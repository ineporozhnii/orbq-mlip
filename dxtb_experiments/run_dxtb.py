import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import load_xyz, work_lmdb, calculate_dxtb 
import multiprocessing
import sys

def main():
    scp_mode = int(sys.argv[1])
    dir_path = str(sys.argv[2])
    save_dir = str(sys.argv[3])
    init_guess_file = str(sys.argv[4])
    n_processes = multiprocessing.cpu_count()
    
    assert os.path.exists(dir_path), f"Data directory {dir_path} does not exist!"
    assert os.path.exists(save_dir), f"Save directory {save_dir} does not exist!"
    
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
            npz_files = [npz_path for npz_path in npz_files if not os.path.exists(f"{save_dir}/{os.path.basename(npz_path).split('.')[0]}_dxtb_result.npz")]
            molecule_ids = [os.path.basename(npz_path).split(".")[0] for npz_path in npz_files]
            args_list = [(str(npz_path), molecule_id, save_dir, scp_mode, None) for (npz_path, molecule_id) in zip(npz_files, molecule_ids)]
   
            print(f"Running parallel calculations with {n_processes} processes...")
            print(args_list[0])
            with multiprocessing.Pool(n_processes) as pool:
                results = list(tqdm(pool.imap(calculate_dxtb, args_list), total=len(args_list)))            
            
    
    # If xyz file is provided
    if filename is not None:
        assert filename.endswith(".xyz"), "File must be an XYZ file."
        assert os.path.exists(filename), f"File {filename} does not exist!"
        molecule_id = os.path.basename(filename).split(".")[0]
        try:
            numbers, positions = load_xyz(filename)
            calculate_dxtb(numbers, positions, molecule_id, save_dir)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # If LMDB file is provided              
    if lmbd_path is not None:
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