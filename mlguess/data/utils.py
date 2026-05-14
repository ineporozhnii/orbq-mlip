import io
import torch
import numpy as np

def read_pt(filecontent):
    if isinstance(filecontent, bytes):
        data = torch.load(io.BytesIO(filecontent), map_location="cpu", weights_only=False)
    else:
        data = torch.load(filecontent, map_location="cpu", weights_only=False)

    atomic_numbers = data["atom_types"].detach().clone().long()
    cart_coords = data["coords"].detach().clone().float()
    
    total_energy = data.get("total_energy")
    total_energy = total_energy.detach().clone().float()
    
    orbital_charges_dxtb = data.get("dxtb_charges_orbital")
    orbital_charges_dxtb = orbital_charges_dxtb.detach().clone().float()
    
    atomic_charges_dxtb = data.get("dxtb_charges_atomic")
    if atomic_charges_dxtb is not None:
        atomic_charges_dxtb = torch.zeros_like(atomic_numbers, dtype=torch.float32)
    atomic_charges_dxtb = atomic_charges_dxtb.detach().clone().float()
    
    atomic_charges_mulliken = data.get("mulliken_charges")
    if atomic_charges_mulliken is not None:
        atomic_charges_mulliken = torch.zeros_like(atomic_numbers, dtype=torch.float32)
    atomic_charges_mulliken = atomic_charges_mulliken.detach().clone().float()
    
    atomic_charges_lowdin = data.get("lowdin_charges")
    if atomic_charges_lowdin is not None:
        atomic_charges_lowdin = torch.zeros_like(atomic_numbers, dtype=torch.float32)
    atomic_charges_lowdin = atomic_charges_lowdin.detach().clone().float()
        
    cell = torch.tensor([[100, 0, 0],
                         [0, 100, 0],
                         [0, 0, 100]]).detach().clone().float()
                         
    if orbital_charges_dxtb.dim() == 2:
        raise ValueError("orbital_charges_dxtb should be a 1D tensor of pre-extracted orbital blocks. Please regenerate dataset.")
        
    forces = data.get("forces")
    if forces is not None:
        forces = forces.detach().clone().float()

    return atomic_numbers, cart_coords, cell, total_energy, orbital_charges_dxtb, atomic_charges_mulliken, atomic_charges_lowdin, atomic_charges_dxtb, forces

def make_graph(
    cell,
    coords_src,
    coords_dst,
    cutoff=4.0,
    disable_pbc=False,
    disable_sc=False,
    max_neigh=None,
):
    """
    compute edges between coords_src and coords_dst under PBC.
    coords_dst is meant to be used for the probes too, which is dense. Therefore, 
    This function may become memory intensive. 
    <max_n_node> is the maximum number of nodes to split <coords_dst>
    into multiple chunks so that we can run this on GPU.
    
    Args:
    """
    if disable_pbc:
        repeat_offsets = torch.tensor([[0, 0, 0]]).to(cell.device)
        shifts = torch.tensor([[0, 0, 0]], dtype=float).to(cell.device)
    else:
        raise NotImplementedError("PBC is not implemented yet.")
        
    # m x supercell x 3. minus shifts because atoms are sources
    shifted = coords_src.unsqueeze(1) - shifts.unsqueeze(0)
    # m x 1 x 1 x 3 - 1 x n x supercell x 3
    diff_vec = coords_dst.unsqueeze(1).unsqueeze(1) - shifted.unsqueeze(0)
    # m x n x supercell
    dist_sqr = diff_vec.pow(2).sum(dim=-1)
    mask = torch.le(dist_sqr, cutoff * cutoff)

    if dist_sqr.shape[0] == dist_sqr.shape[1] and disable_sc:
        mask = torch.logical_and(mask, torch.gt(dist_sqr, 1e-6))
        # remove self loops from topk
        dist_sqr[torch.le(dist_sqr, 1e-6)] = 100.
    
    if max_neigh is not None:
        _, topk_indices = dist_sqr.flatten(1,2).topk(max_neigh, dim=1, largest=False)
        topk_mask = torch.zeros_like(dist_sqr, dtype=torch.bool).flatten(1,2)
        topk_mask.scatter_(1, topk_indices, True)
        topk_mask = topk_mask.view(dist_sqr.shape)
        mask = torch.logical_and(mask, topk_mask)
            
    # dst - src in diff calc  
    idx_dst, idx_src, shift_idx = torch.nonzero(
        mask, as_tuple=True
    )
    unit_shifts = repeat_offsets[shift_idx]
    shifts = torch.matmul(unit_shifts.detach().clone().float(), cell)

    return idx_src, idx_dst, shifts, unit_shifts