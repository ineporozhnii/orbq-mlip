import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import load_xyz, work_lmdb, calculate_dxtb 
import sys

def main():
    scp_mode = int(sys.argv[1])
    dir_path = str(sys.argv[2])
    save_dir = str(sys.argv[3])
    init_guess_file = str(sys.argv[4])
    
    if init_guess_file:
        assert os.path.exists(init_guess_file), f"!!! Init guess file {init_guess_file} does not exist !!!"
        init_guess = np.load(init_guess_file)
        print("")
        print(f"Using guess file: {init_guess_file}")
        print("")
    else:
        init_charges = None
    
    ftype = "npz"
    filename = None
    lmbd_path = None
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
                if "-" in molecule_id or "dsgdb9nsd" in molecule_id:
                    continue
                if os.path.exists(f"{save_dir}/{molecule_id}_dxtb_result.npz"):
                    continue
                if init_guess_file:
                    if molecule_id in init_guess.keys():
                        init_charges = init_guess[molecule_id]
                    else:
                        continue
                else:
                    try:
                        data = np.load(npz_path)
                        numbers = data["atomic_numbers"]
                        positions = data["positions"]
                        calculate_dxtb(numbers, positions, molecule_id, save_dir, scp_mode, init_charges)
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