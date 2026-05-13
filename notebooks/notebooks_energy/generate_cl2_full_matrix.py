import os
import io
import json
import math
import torch
import numpy as np
import dxtb
from tad_mctc.units import AA2AU
from dxtb._src.scf.iterator import get_refocc

def get_random_rotation_matrix():
    angle_x = torch.rand(1).item() * 2 * math.pi
    angle_y = torch.rand(1).item() * 2 * math.pi
    angle_z = torch.rand(1).item() * 2 * math.pi
    
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(angle_x), -math.sin(angle_x)], [0.0, math.sin(angle_x), math.cos(angle_x)]], dtype=torch.float64)
    ry = torch.tensor([[math.cos(angle_y), 0.0, math.sin(angle_y)], [0.0, 1.0, 0.0], [-math.sin(angle_y), 0.0, math.cos(angle_y)]], dtype=torch.float64)
    rz = torch.tensor([[math.cos(angle_z), -math.sin(angle_z), 0.0], [math.sin(angle_z), math.cos(angle_z), 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    
    return rx @ ry @ rz

def generate_cl2_full_matrix_dataset():
    out_dir = "/home/ihor/projects/MLGuessQC/datasets/Cl2_debug_full_matrix_charge/Cl2_dxtb_results"
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
        
        energy = res.total.sum().item()
        
        npz_dict = {
            "numbers": atomic_numbers.numpy().astype(np.int32),
            "coord": cart_coord_rot.numpy().astype(np.float32),
            "final_charges": Q_true.cpu().numpy().astype(np.float32),
            "total_energy_corrected": energy,
            "mulliken_charges": np.zeros_like(atomic_numbers),
            "lowdin_charges": np.zeros_like(atomic_numbers),
            "dxtb_charges": np.zeros_like(atomic_numbers),
        }
        
        filename = f"{str(i).zfill(6)}.npz"
        np.savez(os.path.join(out_dir, filename), **npz_dict)
    
    splits = {
        "train": [f"{str(i).zfill(6)}.npz" for i in range(1)],
        "val": [f"{str(i).zfill(6)}.npz" for i in range(1, 11)],
        "test": [f"{str(i).zfill(6)}.npz" for i in range(11, 21)]
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
