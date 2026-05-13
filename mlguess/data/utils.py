import io
import torch
import numpy as np
import dxtb
from dxtb_experiments.utils import dxtb_to_escn_orbitals
from tad_mctc.units import AA2AU

def read_npz(filecontent):
    data = np.load(io.BytesIO(filecontent))
    atomic_numbers = torch.from_numpy(data["numbers"]).long()
    cart_coords = torch.from_numpy(data["coord"]).float() / AA2AU
    orbital_charges_dxtb = torch.from_numpy(data["final_charges"]).float()
    try:
        total_energy = torch.tensor(data['total_energy_corrected']).float()
        
        atomic_charges_mulliken = torch.from_numpy(data["mulliken_charges"]).float()
        atomic_charges_lowdin = torch.from_numpy(data["lowdin_charges"]).float()
        atomic_charges_dxtb = torch.from_numpy(data["dxtb_charges"]).float()
    except Exception as e:
        print(f"Warning: Missing keys in npz file. Setting default values. Error: {e}")
        total_energy = torch.tensor(0.0).float()
        
        atomic_charges_mulliken = torch.zeros_like(atomic_numbers).float()
        atomic_charges_lowdin = torch.zeros_like(atomic_numbers).float()
        atomic_charges_dxtb = torch.zeros_like(atomic_numbers).float()
    cell = torch.tensor([[100, 0, 0],
                         [0, 100, 0],
                         [0, 0, 100]]).float()
    
    calc = dxtb.calculators.GFN1Calculator(atomic_numbers)
    
    if orbital_charges_dxtb.dim() == 1:
        # Standard 1D populations
        orbital_charges_dxtb = dxtb_to_escn_orbitals(
            angular_momentum_per_shell=calc.ihelp.angular,
            property_tensor=orbital_charges_dxtb,
            map_shell_to_atom_idx=calc.ihelp.shells_to_atom,
            map_orbital_to_shell_idx=calc.ihelp.orbitals_to_shell
        )
    elif orbital_charges_dxtb.dim() == 2:
        # Full 2D Density Matrix
        n_diagonals = orbital_charges_dxtb.shape[0]
        dummy_indices = torch.arange(n_diagonals, dtype=torch.float)
        
        sort_perm = dxtb_to_escn_orbitals(
            angular_momentum_per_shell=calc.ihelp.angular,
            property_tensor=dummy_indices,
            map_shell_to_atom_idx=calc.ihelp.shells_to_atom,
            map_orbital_to_shell_idx=calc.ihelp.orbitals_to_shell
        ).long()
        
        # Sort rows and columns
        P_sorted = orbital_charges_dxtb[sort_perm][:, sort_perm]
        
        # Extract blocks for each shell per atom
        blocks = []
        offset = 0
        sorted_lm = calc.ihelp.angular[calc.ihelp.orbitals_to_shell][sort_perm]
        sorted_atom = calc.ihelp.shells_to_atom[calc.ihelp.orbitals_to_shell][sort_perm]
        
        for atom_idx in range(len(atomic_numbers)):
            atom_mask = (sorted_atom == atom_idx)
            lms_for_atom = sorted_lm[atom_mask]
            
            if lms_for_atom.numel() > 0:
                unique_lms, counts = torch.unique_consecutive(lms_for_atom, return_counts=True)
                for size in counts:
                    block = P_sorted[offset:offset+size, offset:offset+size]
                    blocks.append(block.flatten())
                    offset += size
                    
        orbital_charges_dxtb = torch.cat(blocks)

    return atomic_numbers, cart_coords, cell, total_energy, orbital_charges_dxtb, \
            atomic_charges_mulliken, atomic_charges_lowdin, atomic_charges_dxtb


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
    shifts = torch.matmul(unit_shifts.float(), cell)

    return idx_src, idx_dst, shifts, unit_shifts