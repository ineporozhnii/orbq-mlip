"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Adapted from:
https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core/models/scn
"""

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

from mlguess.model.scn.sampling import CalcSpherePoints

from e3nn import o3
from e3nn.o3 import (
    ReducedTensorProducts,
    Irrep,
    Irreps,
    Linear,
    FullyConnectedTensorProduct,
)


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
        regress_forces: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.regress_forces = regress_forces

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
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels
        self.basis_width_scalar = basis_width_scalar
        self.distance_function = distance_function
        self.device = torch.cuda.current_device()

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
            for m in range(max(self.lmax_list) + 1):
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

        L_max = max(self.lmax_list)
        irreps_in_list = []
        for l in range(L_max + 1):
            parity = "e" if l % 2 == 0 else "o"
            irreps_in_list.append(f"{self.sphere_channels}x{l}{parity}")

        irreps_in = " + ".join(irreps_in_list)

        self.use_equivariant_predictor = True

        if self.use_equivariant_predictor:
            self.lin_out = EquivariantPopulationPredictor(
                irreps_in, max_n_orbitals_per_L=self.max_n_orbitals_per_L
            )
        else:
            self.lin_out = InvariantPopulationPredictor(
                irreps_in, max_n_orbitals_per_L=self.max_n_orbitals_per_L
            )

        self.energy_block = EnergyBlock(
            self.sphere_channels_all, self.num_sphere_samples, self.act
        )
        if getattr(self, "regress_forces", False):
            self.force_block = ForceBlock(
                self.sphere_channels_all, self.num_sphere_samples, self.act
            )

        self.atomic_charge_block = AtomicChargeBlock(
            self.sphere_channels_all, self.num_sphere_samples, self.act
        )

        # self.atomic_charge_lin = o3.Linear(irreps_in=irreps_in, irreps_out="1x0e")

        # Create a roughly evenly distributed point sampling of the sphere for the output blocks
        self.sphere_points = CalcSpherePoints(
            self.num_sphere_samples, self.device
        ).detach()

        # For each spherical point, compute the spherical harmonic coefficient weights
        self.sphharm_weights = []
        for i in range(self.num_resolutions):
            self.sphharm_weights.append(
                o3.spherical_harmonics(
                    torch.arange(0, self.lmax_list[i] + 1).tolist(),
                    self.sphere_points,
                    False,
                ).detach()
            )

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
            self.SO3_edge_rot.append(SO3_Rotation(edge_rot_mat, self.lmax_list[i]))

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
            x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                :, offset : offset + self.sphere_channels
            ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(self.lmax_list, self.mmax_list, device)

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

        ###############################################################
        # Predicts orbital charges
        ###############################################################

        x_emb_native = x.embedding
        node_emb = x_emb_native[:, 0, :]

        e3nn_input_parts = []
        current_lm_idx = 0
        L_max = max(self.lmax_list)
        for l_val in range(L_max + 1):
            num_m_components = (2 * l_val) + 1
            l_block_native = x_emb_native[
                :, current_lm_idx : current_lm_idx + num_m_components, :
            ]
            l_block_e3nn_ordered = l_block_native.permute(0, 2, 1).contiguous()
            e3nn_input_parts.append(
                l_block_e3nn_ordered.reshape(x_emb_native.size(0), -1)
            )
            current_lm_idx += num_m_components

        x_pt_for_e3nn = torch.cat(e3nn_input_parts, dim=1)
        orbital_charges = self.lin_out(x_pt_for_e3nn)

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        x_pt_energy = torch.tensor([], device=device)
        offset = 0
        # Compute the embedding values at every sampled point on the sphere
        for i in range(self.num_resolutions):
            num_coefficients = int((x.lmax_list[i] + 1) ** 2)
            x_pt_energy = torch.cat(
                [
                    x_pt_energy,
                    torch.einsum(
                        "abc, pb->apc",
                        x_emb_native[:, offset : offset + num_coefficients],
                        self.sphharm_weights[i],
                    ).contiguous(),
                ],
                dim=2,
            )
            offset = offset + num_coefficients

        x_pt_energy = x_pt_energy.view(-1, self.sphere_channels_all)

        ###############################################################
        # Energy prediction
        ###############################################################
        node_energy = self.energy_block(x_pt_energy)
        energy = torch.zeros(len(data.n_atom), device=device)
        energy.index_add_(0, data.batch, node_energy.view(-1))

        ################################################################
        # Atomic Charge prediction
        ################################################################
        node_charges = self.atomic_charge_block(x_pt_energy).flatten()

        if getattr(self, "regress_forces", False):
            forces = self.force_block(x_pt_energy, self.sphere_points)
            return energy, forces, orbital_charges, node_charges, node_emb
        else:
            return energy, orbital_charges, node_charges, node_emb

    # Initialize the edge rotation matrices
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        device = edge_distance_vec.device
        dtype = edge_distance_vec.dtype
        norm_eps = 1e-8  # Small epsilon for normalization stability

        # edge_distance_vec are the raw edge vectors r_j - r_i
        edge_lengths = torch.norm(edge_distance_vec, p=2, dim=1, keepdim=True)

        # Normalized edge vector, this will be our local Z-axis
        axis_z = edge_distance_vec / (edge_lengths + norm_eps)
        world_z_torch = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        world_x_torch = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

        # Expand world vectors to match batch size of axis_z
        world_z_vec = world_z_torch.expand_as(axis_z)
        world_x_vec = world_x_torch.expand_as(axis_z)

        dot_with_world_z = torch.sum(axis_z * world_z_vec, dim=1, keepdim=True)

        ref_vec_for_y = torch.where(
            torch.abs(dot_with_world_z) >= (1.0 - norm_eps),  # If aligned with world_z
            world_x_vec,  # Use world_x
            world_z_vec,  # Else use world_z
        )

        axis_y = -torch.cross(
            ref_vec_for_y, axis_z, dim=1
        )  # Swapped order to get Y before X
        axis_y = axis_y / (torch.norm(axis_y, p=2, dim=1, keepdim=True) + norm_eps)

        axis_x = torch.cross(axis_y, axis_z, dim=1)  # axis_y x axis_z = axis_x

        # Columns of edge_rot_mat_inv are axis_x, axis_y, axis_z
        edge_rot_mat_inv = torch.stack([axis_x, axis_y, axis_z], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2).contiguous()

        # RETURN DIRECTLY - Do not permute the geometric coordinate frame
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
            num_channels_m0 = num_channels_m0 + num_coefficents * self.sphere_channels

        # SO(2) convolution for m=0
        self.fc1_dist0 = nn.Linear(edge_channels, self.hidden_channels)
        self.fc1_m0 = nn.Linear(num_channels_m0, self.hidden_channels, bias=False)
        self.fc2_m0 = nn.Linear(self.hidden_channels, num_channels_m0, bias=False)

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
            x.embedding[:, offset : offset + 2 * mappingReduced.m_size[m]] = x_m

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

            num_channels = num_channels + num_coefficents * self.sphere_channels

        assert num_channels > 0

        # Embedding function of the distance
        self.fc1_dist = nn.Linear(edge_channels, 2 * self.hidden_channels)

        # Real weights of SO(2) convolution
        self.fc1_r = nn.Linear(num_channels, self.hidden_channels, bias=True)  # True
        self.fc2_r = nn.Linear(self.hidden_channels, num_channels, bias=False)

        # Imaginary weights of SO(2) convolution
        self.fc1_i = nn.Linear(num_channels, self.hidden_channels, bias=True)  # True
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
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
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


class EnergyBlock(torch.nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels,
        num_sphere_samples,
        act,
    ):
        super(EnergyBlock, self).__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt):
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        node_energy = torch.sum(x_pt, dim=1) / self.num_sphere_samples

        return node_energy


class AtomicChargeBlock(torch.nn.Module):
    """
    Atomic Charge Block: Output block computing the atomic charges

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(
        self,
        num_channels,
        num_sphere_samples,
        act,
    ):
        super(AtomicChargeBlock, self).__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt):
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        # Average over the sphere samples to get the scalar charge per atom
        node_charge = torch.sum(x_pt, dim=1) / self.num_sphere_samples

        return node_charge


class ForceBlock(torch.nn.Module):
    def __init__(self, num_channels, num_sphere_samples, act):
        super(ForceBlock, self).__init__()
        self.num_channels = num_channels
        self.num_sphere_samples = num_sphere_samples
        self.act = act
        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def forward(self, x_pt, sphere_points):
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * sphere_points.view(1, self.num_sphere_samples, 3)
        forces = torch.sum(forces, dim=1) / self.num_sphere_samples
        return forces


class EquivariantPopulationPredictor(nn.Module):
    def __init__(self, irreps_in, max_n_orbitals_per_L):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)

        self.n_s = int(max_n_orbitals_per_L[0]) if len(max_n_orbitals_per_L) > 0 else 0
        self.n_p = int(max_n_orbitals_per_L[1]) if len(max_n_orbitals_per_L) > 1 else 0
        self.n_d = int(max_n_orbitals_per_L[2]) if len(max_n_orbitals_per_L) > 2 else 0

        # --- REGISTER DXTB TRANSFORMATION MATRIX ---
        # Decoded via Overlap Matrix (S) optimization.
        C_matrix = torch.tensor(
            [
                [0.0000, 1.3376, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.3376, 0.0000],
                [0.0000, 0.0000, -0.6688, 0.0000, 1.1584],
                [1.3376, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, -1.1584, 0.0000, -0.6688],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("C_dxtb", C_matrix)
        # -------------------------------------------

        if self.n_s > 0:
            self.predict_s = Linear(self.irreps_in, f"{self.n_s}x0e")

        if self.n_p > 0:
            self.rtp_p = ReducedTensorProducts("ij", i="1o", j="1o")
            self.predict_p = nn.ModuleList(
                [Linear(self.irreps_in, self.rtp_p.irreps_out) for _ in range(self.n_p)]
            )

        if self.n_d > 0:
            self.rtp_d = ReducedTensorProducts("ij", i="2e", j="2e")
            l_max_in = max([l for _, (l, p) in self.irreps_in])
            self.needs_tp = l_max_in < 4  # Will evaluate to True for Lmax=2

            if self.needs_tp:
                # Use a lightweight bottleneck to keep the quadratic tensor product fast
                bottleneck = "16x0e + 16x1o + 16x2e"
                self.proj_down = Linear(self.irreps_in, bottleneck)

                # LINEAR BYPASS: Learn the population trace (0e) and base shapes (2e) linearly
                self.predict_d_lin = nn.ModuleList(
                    [
                        Linear(self.irreps_in, "1x0e + 1x1e + 1x2e")
                        for _ in range(self.n_d)
                    ]
                )
                # QUADRATIC PATH: Synthesize the missing 4e components via Tensor Product
                self.predict_d_tp = nn.ModuleList(
                    [
                        FullyConnectedTensorProduct(
                            bottleneck, bottleneck, "1x3e + 1x4e"
                        )
                        for _ in range(self.n_d)
                    ]
                )
            else:
                self.predict_d = nn.ModuleList(
                    [
                        Linear(self.irreps_in, self.rtp_d.irreps_out)
                        for _ in range(self.n_d)
                    ]
                )

    def forward(self, x):
        outputs = []

        if self.n_s > 0:
            outputs.append(self.predict_s(x))

        if self.n_p > 0:
            for lin in self.predict_p:
                p_tensor = lin(x)
                p_matrix = torch.einsum(
                    "nc, cij -> nij", p_tensor, self.rtp_p.change_of_basis.to(x.device)
                )
                outputs.append(p_matrix.flatten(start_dim=1))

        if self.n_d > 0:
            if self.needs_tp:
                x_proj = self.proj_down(x)

            for i in range(self.n_d):
                if self.needs_tp:
                    d_tensor_lin = self.predict_d_lin[i](x)
                    d_tensor_tp = self.predict_d_tp[i](x_proj, x_proj)
                    # Concatenate to form the full irreps_out (0e + 2e + 4e)
                    d_tensor = torch.cat([d_tensor_lin, d_tensor_tp], dim=1)
                else:
                    d_tensor = self.predict_d[i](x)

                # 1. Reconstruct e3nn's native 5x5 symmetric matrix
                d_matrix_e3nn = torch.einsum(
                    "nc, cij -> nij", d_tensor, self.rtp_d.change_of_basis.to(x.device)
                )

                # 2. Match mixed-precision dtypes
                C = self.C_dxtb.type_as(d_matrix_e3nn)

                # 3. Apply the Cartesian transformation: M_dxtb = C * M_e3nn * C^T
                C_inv = torch.linalg.inv(C)
                d_matrix_dxtb = C @ d_matrix_e3nn @ C_inv

                # 4. Extract the exact equivariant populations!
                pop_d_dxtb = d_matrix_dxtb.flatten(start_dim=1)
                outputs.append(pop_d_dxtb)

        return torch.cat(outputs, dim=1)


class InvariantPopulationPredictor(nn.Module):
    def __init__(self, irreps_in, max_n_orbitals_per_L):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)

        self.n_s = int(max_n_orbitals_per_L[0]) if len(max_n_orbitals_per_L) > 0 else 0
        self.n_p = int(max_n_orbitals_per_L[1]) if len(max_n_orbitals_per_L) > 1 else 0
        self.n_d = int(max_n_orbitals_per_L[2]) if len(max_n_orbitals_per_L) > 2 else 0

        # Calculate the total number of individual orbital charges required by the basis set.
        # s-shell = 1 scalar, p-shell = 9 scalars (3x3), d-shell = 25 scalars (5x5)
        self.total_scalar_outputs = self.n_s + (self.n_p * 9) + (self.n_d * 25)

        # e3nn's Linear automatically maps the 0e inputs to 0e outputs.
        # It rigorously zeroes out any connection to the 1o, 2e, 3o, and 4e features.
        self.predict_invariant = Linear(
            self.irreps_in, f"{self.total_scalar_outputs}x0e"
        )

    def forward(self, x):
        # x contains the full geometric embedding.
        # The output is a flat tensor of shape [batch_size, total_scalar_outputs]
        return self.predict_invariant(x)
