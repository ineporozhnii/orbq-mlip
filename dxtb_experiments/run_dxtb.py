import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import load_xyz, work_lmdb, calculate_dxtb 

def main():
    
    ftype = "npz"
    filename = None
    dir_path = None
    lmbd_path = None
    save_dir = None
    #filename = "all_molecules.lmdb"
    #filename = "tmqm_delta_e_neutral_ref_cor_70_test.lmdb"  # Replace with your file path
    #filename = "/home/ihor/projects/MLGuessQC/data/JAXREF.xyz"
    
    # If directory is provided
    if dir_path is not None:
        if ftype == "xyz":
            xyz_files = list(Path(dir_path).glob("*.xyz"))
            for xyz_path in tqdm(xyz_files):
                molecule_id = os.path.basename(xyz_path).split(".")[0]
                if os.path.exists(f"{save_dir}/{molecule_id}_dxtb_result.npz"):
                    continue
                else:
                    try:
                        numbers, positions = load_xyz(str(xyz_path))
                        calculate_dxtb(numbers, positions, molecule_id, save_dir)
                    except Exception as e:
                        print(f"Error processing {xyz_path}: {e}")
        elif ftype == "npz":
            npz_files = list(Path(dir_path).glob("*.npz"))
            for npz_path in tqdm(npz_files):
                molecule_id = os.path.basename(npz_path).split(".")[0]
                if os.path.exists(f"{save_dir}/{molecule_id}_dxtb_result.npz"):
                    continue
                if 'dsgdb9nsd' in molecule_id:
                    continue  # Skip dsgdb9nsd molecules
                else:
                    try:
                        data = np.load(npz_path)
                        numbers = data["atomic_numbers"]
                        positions = data["positions"]
                        calculate_dxtb(numbers, positions, molecule_id, save_dir)
                    except Exception as e:
                        print(f"Error processing {npz_path}: {e}")
            
    
    # If xyz file is provided
    if filename.endswith(".xyz"):
        molecule_id = os.path.basename(filename).split(".")[0]
        try:
            numbers, positions = load_xyz(filename)
            calculate_dxtb(numbers, positions, molecule_id, save_dir)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # If LMDB file is provided              
    if filename.endswith(".lmdb"):
        dataset = work_lmdb(lmbd_path, filename)
        for i in tqdm(len(dataset)):
            molecule = dataset[i]
            molecule_id = molecule.cid
            numbers = molecule["atomic_numbers"]
            positions = molecule["pos"]
            if os.path.exists(f"{save_dir}/{molecule_id}_dxtb_result.npz"):
                continue
            else:
                try:
                    calculate_dxtb(numbers, positions, molecule_id, save_dir)
                except Exception as e:
                    print(f"Error processing {molecule_id}: {e}")
    
if __name__ == "__main__":
    main()