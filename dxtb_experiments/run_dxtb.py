import os
from tqdm import tqdm
from pathlib import Path
from utils import load_xyz, work_lmdb, calculate_dxtb 

def main():
    N_MAX = 5000
    filename = ""
    lmbd_path = ""
    save_dir = ""
    #filename = "all_molecules.lmdb"
    #filename = "tmqm_delta_e_neutral_ref_cor_70_test.lmdb"  # Replace with your file path
    #filename = "/home/ihor/projects/MLGuessQC/data/JAXREF.xyz"
    
    # If directory is provided
    if os.path.isdir(filename):
        # Get all XYZ files in the directory
        xyz_files = list(Path(filename).glob("*.xyz"))
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