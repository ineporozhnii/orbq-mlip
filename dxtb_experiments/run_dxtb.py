import time
import torch
import dxtb
import lmdb
import json
import numpy as np
from ase.io import read
from tqdm import tqdm
from ocpmodels.datasets import LmdbDataset
import id_config as cfg
import os

def work_lmdb(file_path: str, file_name: str) -> LmdbDataset:
    lmdb_env = lmdb.open(file_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    db = lmdb.open(
        file_path+file_name,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
  
    dataset = LmdbDataset({"src": file_path+file_name})
    return dataset


def load_structure(filename: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a structure from an XYZ file.
    """
    # Load the structure from an XYZ file
    atoms = read(filename)

    # Extract atomic numbers and positions
    numbers = torch.tensor([atom.number for atom in atoms], dtype=torch.int64)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.double)

    return numbers, positions

def main():
    # Load the structure from an XYZ file
    filename = "all_molecules.lmdb"
    #filename = "tmqm_delta_e_neutral_ref_cor_70_test.lmdb"  # Replace with your file path
    #filename = "/home/ihor/projects/MLGuessQC/data/JAXREF.xyz"
    if filename.endswith(".xyz"):
        numbers, positions = load_structure(filename)

        # Set device and dtype
        dd = {"dtype": torch.double, "device": torch.device("cpu")}

        # Instantiate a calculator
        calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

        start = time.time()
        result = calc.singlepoint(positions)
        end = time.time()
        calc_time = end - start
        print(f"Calculation time: {calc_time:.2f} seconds")
        n_scf_iter = np.array([result.iter])
        calc_time = np.array([calc_time])
        final_charges = result.charges.mono.detach().numpy()
        final_emo = result.emo.detach().numpy()
        final_energy = result.total.detach().numpy()
        #init_eeq_charges = torch.load("dxtb/temp/eeq_charges.pt").detach().numpy()
        
        # np.savez("dxtb/results/9393_dxtb_result", 
        #          init_eeq_charges=init_eeq_charges, 
        #          final_charges=final_charges,  
        #          final_emo=final_emo, 
        #          final_energy=final_energy,
        #          n_scf_iter=n_scf_iter,
        #          calc_time=calc_time)
    
    elif filename.endswith(".lmdb"):
        dataset = work_lmdb("/home/ihor/projects/MLGuessQC/data/quinones/", filename)
        # Load the structure from the LMDB file
        for i in tqdm(range(0, 5000)):
            molecule = dataset[i]
            numbers = molecule["atomic_numbers"].clone().detach().to(torch.int64)
            positions = molecule["pos"].clone().detach().to(torch.double).requires_grad_(True)
            # Set device and dtype
            dd = {"dtype": torch.double, "device": torch.device("cpu")}
            # Instantiate a calculator
            cfg.CURRENT_MOLECULE_ID = molecule.cid
            calc = dxtb.calculators.GFN1Calculator(numbers, **dd, opts={"verbosity": 3})
            
            start = time.time()
            result = calc.singlepoint(positions)
            end = time.time()
            calc_time = end - start
            
            n_scf_iter = np.array([result.iter])
            calc_time = np.array([calc_time])
            final_charges = result.charges.mono.detach().numpy()
            final_emo = result.emo.detach().numpy()
            final_energy = result.total.detach().numpy()
            init_eeq_charges = torch.load("dxtb_experiments/temp/eeq_charges.pt").detach().numpy()
            with open("dxtb_experiments/temp/qdiff.json", "r") as file:
                data = json.load(file)
                qdiff = np.array(data["qdiff_values"])
            os.remove("dxtb_experiments/temp/qdiff.json")
            
            np.savez(f"dxtb_experiments/results_quinones_noisy_guess_0p2std/{molecule.cid}_dxtb_result", 
                 init_eeq_charges=init_eeq_charges, 
                 final_charges=final_charges,  
                 final_emo=final_emo, 
                 final_energy=final_energy,
                 n_scf_iter=n_scf_iter,
                 calc_time=calc_time,
                 qdiff=qdiff)
    
if __name__ == "__main__":
    main()