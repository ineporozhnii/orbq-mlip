import os
import io
import json
import math
import torch
import numpy as np
import dxtb
from tad_mctc.units import AA2AU
from dxtb._src.scf.iterator import get_refocc
from dxtb_experiments.utils import dxtb_to_escn_orbitals

def get_random_rotation_matrix():
    angle_x = torch.rand(1).item() * 2 * math.pi
    angle_y = torch.rand(1).item() * 2 * math.pi
    angle_z = torch.rand(1).item() * 2 * math.pi
    
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(angle_x), -math.sin(angle_x)], [0.0, math.sin(angle_x), math.cos(angle_x)]], dtype=torch.float64)
    ry = torch.tensor([[math.cos(angle_y), 0.0, math.sin(angle_y)], [0.0, 1.0, 0.0], [-math.sin(angle_y), 0.0, math.cos(angle_y)]], dtype=torch.float64)
    rz = torch.tensor([[math.cos(angle_z), -math.sin(angle_z), 0.0], [math.sin(angle_z), math.cos(angle_z), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    
    return rx @ ry @ rz

def generate_cl2_full_matrix_dataset():
    out_dir = "/home/ihor/projects/MLGuessQC/datasets/Cl2_debug_full_matrix_charge_forces/Cl2_dxtb_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Base Cl2 coordinates
    atomic_numbers = torch.tensor([17, 17], dtype=torch.long)
    base_coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.99]
    ], dtype=torch.float64)
    
    calc_config = {
        "device": torch.device("cpu"),
        "dtype": torch.float64
    }
    
    num_mols = 21
    
    for i in range(num_mols):
        R = get_random_rotation_matrix()
        cart_coord_rot = base_coords @ R.T
        
        # dxtb expects Bohr
        cart_coord_rot_au = cart_coord_rot * AA2AU
        cart_coord_rot_au = cart_coord_rot_au.requires_grad_(True)
        
        calc = dxtb.calculators.GFN1Calculator(atomic_numbers, **calc_config)
        res = calc.singlepoint(cart_coord_rot_au)
        
        P_true = res.density.squeeze()
        S_true = calc.integrals.build_overlap(cart_coord_rot_au).squeeze()
        M_true = P_true @ S_true
        
        n0, occ = get_refocc(calc.integrals.hcore.refocc, torch.tensor([0.0]), torch.tensor([0.0]), calc.ihelp)
        N0_mat = torch.diag(n0.squeeze())
        
        # Q_true is exactly the orbital charge matrix (Charge = N0 - Population)
        Q_true = N0_mat - M_true
        
        # Verify diagonal matches orbital charges
        mono_charges = res.charges.mono.squeeze()
        assert torch.allclose(torch.diagonal(Q_true), mono_charges, atol=1e-4), "Diagonals of Q do not match orbital charges!"
        
        # Sort and extract blocks
        n_diagonals = Q_true.shape[0]
        dummy_indices = torch.arange(n_diagonals, dtype=torch.float)
        
        sort_perm = dxtb_to_escn_orbitals(
            angular_momentum_per_shell=calc.ihelp.angular,
            property_tensor=dummy_indices,
            map_shell_to_atom_idx=calc.ihelp.shells_to_atom,
            map_orbital_to_shell_idx=calc.ihelp.orbitals_to_shell
        ).long()
        
        Q_sorted = Q_true[sort_perm][:, sort_perm]
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
                    block = Q_sorted[offset:offset+size, offset:offset+size]
                    blocks.append(block.flatten())
                    offset += size
                    
        final_charges = torch.cat(blocks)
        energy = res.total.sum().item()
        
        # Compute exact dxtb forces for debugging dataset
        forces_au = calc.get_forces(cart_coord_rot_au)
        forces = forces_au / AA2AU  # Convert back to atomic units per Angstrom, or whatever units used
        
        data_dict = {
            "atom_types": atomic_numbers.long(),
            "coords": cart_coord_rot.float(),
            "dxtb_charges_orbital": final_charges.cpu().float(),
            "total_energy": torch.tensor(energy, dtype=torch.float32),
            "dxtb_charges_atomic": torch.zeros_like(atomic_numbers, dtype=torch.float32),
            "mulliken_charges": torch.zeros_like(atomic_numbers, dtype=torch.float32),
            "lowdin_charges": torch.zeros_like(atomic_numbers, dtype=torch.float32),
            "forces": forces.cpu().float(),
        }
        
        filename = f"{str(i).zfill(6)}.pt"
        torch.save(data_dict, os.path.join(out_dir, filename))
    
    splits = {
        "train": [f"{str(i).zfill(6)}.pt" for i in range(1)],
        "val": [f"{str(i).zfill(6)}.pt" for i in range(1, 11)],
        "test": [f"{str(i).zfill(6)}.pt" for i in range(11, 21)]
    }
    with open(os.path.join(out_dir, "datasplits.json"), "w") as f:
        json.dump(splits, f, indent=4)
        
    metadata = {
        "avg_num_neighbors": 1.0,
        "energy_var": 1.0,
        "atomic_charges_dxtb_var": 1.0,
        "atomic_charges_lowdin_var": 1.0,
        "atomic_charges_mulliken_var": 1.0,
        "orbital_charges_dxtb_var": 1.0,
        "unique_atom_types": [17]
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Generated {num_mols} full-matrix debug molecules in {out_dir}")

if __name__ == "__main__":
    generate_cl2_full_matrix_dataset()
