"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Adapted from: 
https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core/models/scn
"""

import logging
import time
from typing import List

import torch
import torch.nn as nn

from mlguess.common.utils import get_edge_vectors_and_lengths
from mlguess.model.scn.so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
)
from mlguess.model.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from e3nn import o3
from e3nn.nn import BatchNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class eSCN(nn.Module):
    """Equivariant Spherical Channel Network
    Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs

    Args:
        cutoff (float):         Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number
        num_layers (int):             Number of layers in the GNN
        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        hidden_channels (int):        Number of hidden units in message passing
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        edge_channels (int):          Number of channels for the edge invariant features
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float):   Width of distance basis function
        distance_resolution (float):  Distance between distance basis functions in Angstroms
        show_timing_info (bool):      Show timing and memory info
    """

    def __init__(
        self,
        max_n_Ls: torch.Tensor,
        max_n_orbitals_per_L: torch.Tensor,
        cutoff: float = 6.0,
        max_num_elements: int = 100,
        num_layers: int = 8,
        lmax_list: List[int] = [6],
        mmax_list: List[int] = [2],
        sphere_channels: int = 128,
        hidden_channels: int = 256,
        edge_channels: int = 128,
        num_sphere_samples: int = 128,
        distance_function: str = "gaussian",
        basis_width_scalar: float = 1.0,
        distance_resolution: float = 0.02,
        show_timing_info: bool = False,
        expo_trainable: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__()

        self.cutoff = cutoff
        self.show_timing_info = show_timing_info
        self.max_num_elements = max_num_elements
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_sphere_samples = num_sphere_samples
        self.sphere_channels = sphere_channels
        self.edge_channels = edge_channels
        self.distance_resolution = distance_resolution
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = (
            self.num_resolutions * self.sphere_channels
        )
        self.basis_width_scalar = basis_width_scalar
        self.distance_function = distance_function

        # variables used for display purposes
        self.counter = 0

        # non-linear activation function used throughout the network
        self.act = nn.SiLU()

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
            "sigmoid",
            "linearsigmoid",
            "silu",
        ]
        self.num_gaussians = int(cutoff / self.distance_resolution)
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "sigmoid":
            self.distance_expansion = SigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "linearsigmoid":
            self.distance_expansion = LinearSigmoidSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )
        if self.distance_function == "silu":
            self.distance_expansion = SiLUSmearing(
                0.0,
                cutoff,
                self.num_gaussians,
                basis_width_scalar,
            )

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = nn.ModuleList()
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.mmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(lval, m))

            self.SO3_grid.append(SO3_m_grid)

        # Initialize the blocks for each layer of the GNN
        self.layer_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = LayerBlock(
                i,
                self.sphere_channels,
                self.hidden_channels,
                self.edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.distance_expansion,
                self.max_num_elements,
                self.SO3_grid,
                self.act,
            )
            self.layer_blocks.append(block)
            
        self.max_n_orbitals_per_L = max_n_orbitals_per_L
        # assumes len(lmax_list) == 1
        self.expo_trainable = expo_trainable
        
        if max(self.lmax_list) == 1:
            self.coeff_idx = torch.tensor([0, 1, 1*self.sphere_channels, 2*self.sphere_channels, 3*self.sphere_channels]).long()
            self.change_of_basis = torch.tensor([0, 3, 1, 2]).detach()
            
        elif max(self.lmax_list) == 2:
            self.coeff_idx = torch.tensor([0, 1, 1*self.sphere_channels, 2*self.sphere_channels, 3*self.sphere_channels,
                                                 4*self.sphere_channels, 5*self.sphere_channels, 6*self.sphere_channels,
                                                 7*self.sphere_channels, 8*self.sphere_channels]).long()
            self.change_of_basis = torch.tensor([0, 3, 1, 2, 4, 5, 6, 7, 8]).detach()
        else:
            raise NotImplementedError
 
        
    def forward(self, data):
        device = data.coords.device
        self.batch_size = len(data.cell)
        self.dtype = data.coords.dtype
        atomic_numbers = data.atom_types.long()
        num_atoms = len(atomic_numbers)

        start_time = time.time()
        
        edge_index = data["edge_index"]
        edge_distance_vec, edge_distance = get_edge_vectors_and_lengths(
            positions=data["coords"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data.coords, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_edge_rot.append(
                SO3_Rotation(edge_rot_mat, self.lmax_list[i])
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l=0,m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(
                atomic_numbers
            )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(
            self.lmax_list, self.mmax_list, device
        )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            if i > 0:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

                # Residual layer for all layers past the first
                x.embedding = x.embedding + x_message.embedding

            else:
                # No residual for the first layer
                x = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )
            
            #torch.save(x.embedding, f"embedding_layer_{i}.pt")
            
        ###############################################################
        # Predicts orbitals
        ###############################################################
        # N x "Cx0e + Cx1o + Cx2e + Cx3o + ..."
        
        # x_emb_native = x.embedding

        # e3nn_input_parts = []
        # current_lm_idx = 0
        # L_max = max(self.lmax_list)
        # for l_val in range(L_max + 1):
        #     num_m_components = (2 * l_val) + 1
        #     # Select the slice of x_emb_native for this l_val
        #     # These are (N_atoms, num_m_components, S) if SO3_Embedding groups m's for an l consecutively
        #     # It's more likely: x_emb_native[:, current_lm_idx : current_lm_idx + num_m_components, :]
        #     # This assumes that the (L_max+1)**2 dimension is ordered like:
        #     # all m for L=0, then all m for L=1, etc.

        #     l_block_native = x_emb_native[:, current_lm_idx : current_lm_idx + num_m_components, :]
        #     #print(l_block_native.shape)
        #     # l_block_native shape: (N_atoms, num_m_components_for_this_l, S_channels)

        #     # We need to permute to (N_atoms, S_channels, num_m_components_for_this_l)
        #     # so that m varies fastest for each channel.
        #     l_block_e3nn_ordered = l_block_native.permute(0, 2, 1).contiguous()
        #     # Now shape: (N_atoms, S_channels, num_m_components_for_this_l)

        #     e3nn_input_parts.append(l_block_e3nn_ordered.reshape(x_emb_native.size(0), -1))
        #     current_lm_idx += num_m_components

        # x_pt_for_e3nn = torch.cat(e3nn_input_parts, dim=1)

        
        x_emb = x.embedding[:, self.change_of_basis, :]
        coeffs = x_emb.flatten(1,2)[:, self.coeff_idx]

        expo_scaling = None
        return coeffs, expo_scaling
    
    
        # Initialize the edge rotation matrics
    # def _init_edge_rot_mat(self, coords, edge_index, edge_distance_vec):
    #     edge_vec_0 = edge_distance_vec
    #     edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    #     # Make sure the atoms are far enough apart
    #     if torch.min(edge_vec_0_distance) < 0.0001:
    #         logging.error(
    #             "Error edge_vec_0_distance: {}".format(
    #                 torch.min(edge_vec_0_distance)
    #             )
    #         )
    #         (minval, minidx) = torch.min(edge_vec_0_distance, 0)
    #         logging.error(
    #             "Error edge_vec_0_distance: {} {} {} {} {}".format(
    #                 minidx,
    #                 edge_index[0, minidx],
    #                 edge_index[1, minidx],
    #                 coords[edge_index[0, minidx]],
    #                 coords[edge_index[1, minidx]],
    #             )
    #         )

    #     norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    #     edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    #     edge_vec_2 = edge_vec_2 / (
    #         torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    #     )
    #     # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    #     # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    #     edge_vec_2b = edge_vec_2.clone()
    #     edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    #     edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    #     edge_vec_2c = edge_vec_2.clone()
    #     edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    #     edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    #     vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
    #         -1, 1
    #     )
    #     vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
    #         -1, 1
    #     )

    #     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    #     edge_vec_2 = torch.where(
    #         torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    #     )
    #     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    #     edge_vec_2 = torch.where(
    #         torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    #     )

    #     vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    #     # Check the vectors aren't aligned
    #     assert torch.max(vec_dot) < 0.99

    #     norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    #     norm_z = norm_z / (
    #         torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    #     )
    #     norm_z = norm_z / (
    #         torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    #     )
    #     norm_y = torch.cross(norm_x, norm_z, dim=1)
    #     norm_y = norm_y / (
    #         torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    #     )

    #     # Construct the 3D rotation matrix
    #     norm_x = norm_x.view(-1, 3, 1)
    #     norm_y = -norm_y.view(-1, 3, 1)
    #     norm_z = norm_z.view(-1, 3, 1)

    #     edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    #     edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)
        
    #     P = torch.tensor([[0.0, 1.0, 0.0],
    #                       [0.0, 0.0, 1.0],
    #                       [1.0, 0.0, 0.0]], device=edge_rot_mat.device, dtype=edge_rot_mat.dtype)
        
    #     edge_rot_mat = P @ edge_rot_mat @ P.T

    #     return edge_rot_mat.detach()
    

    # def plot_axes(self, axis_x, axis_y, axis_z, id="0"):
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')

    #         # Plot origin
    #         ax.scatter(0, 0, 0, color='black', label='Origin')

    #         # Plot axis vectors
            
    #         ax.quiver(0, 0, 0, 1, 0, 0, color='k')
    #         ax.quiver(0, 0, 0, 0, 1, 0, color='k')
    #         ax.quiver(0, 0, 0, 0, 0, 1, color='k')
            
    #         ax.quiver(0, 0, 0, axis_x[0], axis_x[1], axis_x[2], color='red', label='X-axis')
    #         ax.quiver(0, 0, 0, axis_y[0], axis_y[1], axis_y[2], color='green', label='Y-axis')
    #         ax.quiver(0, 0, 0, axis_z[0], axis_z[1], axis_z[2], color='blue', label='Z-axis')

    #         # Set labels
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')

    #         ax.legend()
    #         plt.savefig(f'axes_plot_original_{id}.png')

    # # Initialize the edge rotation matrics
    
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        device = edge_distance_vec.device
        dtype = edge_distance_vec.dtype
        norm_eps = 1e-8  # Small epsilon for normalization stability

        # edge_distance_vec are the raw edge vectors r_j - r_i
        edge_lengths = torch.norm(edge_distance_vec, p=2, dim=1, keepdim=True)
        
        # Normalized edge vector, this will be our local Z-axis
        axis_z = edge_distance_vec / (edge_lengths + norm_eps)
        # Create a consistent X_local, Y_local
        # Pick a reference vector not collinear with axis_z.
        # Common choice: world Z-axis [0,0,1] or world X-axis [1,0,0].
        # If edge is along world Z, pick world X, else pick world Z.
        world_z_torch = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        world_x_torch = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        P = torch.tensor([[0.0, 1.0, 0.0], 
                          [0.0, 0.0, 1.0], 
                          [1.0, 0.0, 0.0]], device=device, dtype=dtype, requires_grad=False)
        
        # Expand world vectors to match batch size of axis_z
        world_z_vec = world_z_torch.expand_as(axis_z)
        world_x_vec = world_x_torch.expand_as(axis_z)

        dot_with_world_z = torch.sum(axis_z * world_z_vec, dim=1, keepdim=True)

        # If axis_z is (anti)parallel to world_z, use world_x as reference
        # otherwise, use world_z as reference to cross with axis_z to get axis_y
        # (This order makes axis_y = normalize(world_ref x axis_z))
        ref_vec_for_y = torch.where(
            torch.abs(dot_with_world_z) >= (1.0 - norm_eps), # If aligned with world_z
            world_x_vec, # Use world_x
            world_z_vec  # Else use world_z
        )
        
        axis_y = -torch.cross(ref_vec_for_y, axis_z, dim=1) # Swapped order to get Y before X
        axis_y = axis_y / (torch.norm(axis_y, p=2, dim=1, keepdim=True) + norm_eps)

        axis_x = torch.cross(axis_y, axis_z, dim=1) # axis_y x axis_z = axis_x

        # Columns of edge_rot_mat_inv are axis_x, axis_y, axis_z
        edge_rot_mat_inv = torch.stack([axis_x, axis_y, axis_z], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2).contiguous()
        edge_rot_mat = P @ edge_rot_mat @ P.T
        
        return edge_rot_mat.detach()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerBlock(torch.nn.Module):
    """
    Layer block: Perform one layer (message passing and aggregation) of the GNN

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super(LayerBlock, self).__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels = sphere_channels
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid

        # Message block
        self.message_block = MessageBlock(
            self.layer_idx,
            self.sphere_channels,
            hidden_channels,
            edge_channels,
            self.lmax_list,
            self.mmax_list,
            distance_expansion,
            max_num_elements,
            self.SO3_grid,
            self.act,
        )

        # Non-linear point-wise comvolution for the aggregated messages
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc2_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

        self.fc3_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        # Compute messages by performing message block
        x_message = self.message_block(
            x,
            atomic_numbers,
            edge_distance,
            edge_index,
            SO3_edge_rot,
            mappingReduced,
        )

        # Compute point-wise spherical non-linearity on aggregated messages
        max_lmax = max(self.lmax_list)

        # Project to grid
        x_grid_message = x_message.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = x.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = torch.cat([x_grid, x_grid_message], dim=3)

        # Perform point-wise convolution
        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        # Project back to spherical harmonic coefficients
        x_message._from_grid(x_grid, self.SO3_grid, lmax=max_lmax)

        # Return aggregated messages
        return x_message


class MessageBlock(torch.nn.Module):
    """
    Message block: Perform message passing

    Args:
        layer_idx (int):            Layer number
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        layer_idx: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        distance_expansion,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super(MessageBlock, self).__init__()
        self.layer_idx = layer_idx
        self.act = act
        self.hidden_channels = hidden_channels
        self.sphere_channels = sphere_channels
        self.SO3_grid = SO3_grid
        self.num_resolutions = len(lmax_list)
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.edge_channels = edge_channels

        # Create edge scalar (invariant to rotations) features
        self.edge_block = EdgeBlock(
            self.edge_channels,
            distance_expansion,
            max_num_elements,
            self.act,
        )

        # Create SO(2) convolution blocks
        self.so2_block_source = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.so2_block_target = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )

    def forward(
        self,
        x,
        atomic_numbers,
        edge_distance,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        ###############################################################
        # Compute messages
        ###############################################################

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        x_edge = self.edge_block(
            edge_distance,
            atomic_numbers[edge_index[0]],  # Source atom atomic number
            atomic_numbers[edge_index[1]],  # Target atom atomic number
        )

        # Copy embeddings for each edge's source and target nodes
        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        # Rotate the irreps to align with the edge
        x_source._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        x_target._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)

        # Compute messages
        x_source = self.so2_block_source(x_source, x_edge, mappingReduced)
        x_target = self.so2_block_target(x_target, x_edge, mappingReduced)

        # Add together the source and target results
        x_target.embedding = x_source.embedding + x_target.embedding

        # Point-wise spherical non-linearity
        x_target._grid_act(self.SO3_grid, self.act, mappingReduced)

        # Rotate back the irreps
        x_target._rotate_inv(SO3_edge_rot, mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_target._reduce_edge(edge_index[1], len(x.embedding))

        return x_target


class SO2Block(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        act,
    ) -> None:
        super(SO2Block, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(lmax_list)
        self.act = act

        num_channels_m0 = 0
        for i in range(self.num_resolutions):
            num_coefficents = self.lmax_list[i] + 1
            num_channels_m0 = (
                num_channels_m0 + num_coefficents * self.sphere_channels
            )

        # SO(2) convolution for m=0
        self.fc1_dist0 = nn.Linear(edge_channels, self.hidden_channels)
        self.fc1_m0 = nn.Linear(
            num_channels_m0, self.hidden_channels, bias=False
        )
        self.fc2_m0 = nn.Linear(
            self.hidden_channels, num_channels_m0, bias=False
        )

        # SO(2) convolution for non-zero m
        self.so2_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            so2_conv = SO2Conv(
                m,
                self.sphere_channels,
                self.hidden_channels,
                edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.act,
            )
            self.so2_conv.append(so2_conv)

    def forward(
        self,
        x,
        x_edge,
        mappingReduced,
    ):
        num_edges = len(x_edge)

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(mappingReduced)

        # Compute m=0 coefficients separately since they only have real values (no imaginary)

        # Compute edge scalar features for m=0
        x_edge_0 = self.act(self.fc1_dist0(x_edge))

        x_0 = x.embedding[:, 0 : mappingReduced.m_size[0]].contiguous()
        x_0 = x_0.view(num_edges, -1)

        x_0 = self.fc1_m0(x_0)
        x_0 = x_0 * x_edge_0
        x_0 = self.fc2_m0(x_0)
        x_0 = x_0.view(num_edges, -1, x.num_channels)

        # Update the m=0 coefficients
        x.embedding[:, 0 : mappingReduced.m_size[0]] = x_0

        # Compute the values for the m > 0 coefficients
        offset = mappingReduced.m_size[0]
        for m in range(1, max(self.mmax_list) + 1):
            # Get the m order coefficients
            x_m = x.embedding[
                :, offset : offset + 2 * mappingReduced.m_size[m]
            ].contiguous()
            x_m = x_m.view(num_edges, 2, -1)
            # Perform SO(2) convolution
            x_m = self.so2_conv[m - 1](x_m, x_edge)
            x_m = x_m.view(num_edges, -1, x.num_channels)
            x.embedding[
                :, offset : offset + 2 * mappingReduced.m_size[m]
            ] = x_m

            offset = offset + 2 * mappingReduced.m_size[m]

        # Reshape the spherical harmonics based on l (degree)
        x._l_primary(mappingReduced)

        return x


class SO2Conv(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        edge_channels (int):        Size of invariant edge embedding
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        act,
    ) -> None:
        super(SO2Conv, self).__init__()
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.sphere_channels = sphere_channels
        self.num_resolutions: int = len(self.lmax_list)
        self.m = m
        self.act = act

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficents = 0
            if self.mmax_list[i] >= m:
                num_coefficents = self.lmax_list[i] - m + 1

            num_channels = (
                num_channels + num_coefficents * self.sphere_channels
            )

        assert num_channels > 0

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(edge_channels, 2 * self.hidden_channels)

        # Real weights of SO(2) convolution
        self.fc1_r = nn.Linear(num_channels, self.hidden_channels, bias=True) # True 
        self.fc2_r = nn.Linear(self.hidden_channels, num_channels, bias=False)

        # Imaginary weights of SO(2) convolution
        self.fc1_i = nn.Linear(num_channels, self.hidden_channels, bias=True) # True 
        self.fc2_i = nn.Linear(self.hidden_channels, num_channels, bias=False)

    def forward(self, x_m, x_edge) -> torch.Tensor:
        # Compute edge scalar features
        x_edge = self.act(self.fc1_dist(x_edge))
        x_edge = x_edge.view(-1, 2, self.hidden_channels)

        # Perform the complex weight multiplication
        x_r = self.fc1_r(x_m)
        x_r = x_r * x_edge[:, 0:1, :]
        x_r = self.fc2_r(x_r)

        x_i = self.fc1_i(x_m)
        x_i = x_i * x_edge[:, 1:2, :]
        x_i = self.fc2_i(x_i)

        x_m_r = x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r[:, 1] + x_i[:, 0]

        return torch.stack((x_m_r, x_m_i), dim=1).contiguous()


class EdgeBlock(torch.nn.Module):
    """
    Edge Block: Compute invariant edge representation from edge diatances and atomic numbers

    Args:
        edge_channels (int):        Size of invariant edge embedding
        distance_expansion (func):  Function used to compute distance embedding
        max_num_elements (int):     Maximum number of atomic numbers
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        edge_channels,
        distance_expansion,
        max_num_elements,
        act,
    ) -> None:
        super(EdgeBlock, self).__init__()
        self.in_channels = distance_expansion.num_output
        self.distance_expansion = distance_expansion
        self.act = act
        self.edge_channels = edge_channels
        self.max_num_elements = max_num_elements

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(self.in_channels, self.edge_channels)

        # Embedding function of the atomic numbers
        self.source_embedding = nn.Embedding(
            self.max_num_elements, self.edge_channels
        )
        self.target_embedding = nn.Embedding(
            self.max_num_elements, self.edge_channels
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        # Embedding function of the edge
        self.fc1_edge_attr = nn.Linear(
            self.edge_channels,
            self.edge_channels,
        )

    def forward(self, edge_distance, source_element, target_element):
        # Compute distance embedding
        x_dist = self.distance_expansion(edge_distance)
        x_dist = self.fc1_dist(x_dist)

        # Compute atomic number embeddings
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)

        # Compute invariant edge embedding
        x_edge = self.act(source_embedding + target_embedding + x_dist)
        x_edge = self.act(self.fc1_edge_attr(x_edge))

        return x_edge