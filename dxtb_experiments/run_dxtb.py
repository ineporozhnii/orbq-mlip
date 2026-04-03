import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import multiprocessing
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from utils import calculate_dxtb 

def worker_init():
    # Ensure each worker uses exactly 1 thread for PyTorch operations
    torch.set_num_threads(1)

def main():
    # Use spawn for PyTorch multiprocess safety
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    if len(sys.argv) < 4:
        print("Usage: python run_dxtb.py <scp_mode> <dir_path> <save_dir>")
        sys.exit(1)

    scp_mode = int(sys.argv[1])
    dir_path = Path(sys.argv[2])
    save_dir = Path(sys.argv[3])
    n_processes = min(190, multiprocessing.cpu_count())
    
    assert dir_path.exists(), f"Data directory {dir_path} does not exist!"
    assert save_dir.exists(), f"Save directory {save_dir} does not exist!"
    
    ftype = "pt"
    
    args_list = []
    
    # Collect files and filter out already processed ones
    for file_path in dir_path.glob(f"*.{ftype}"):
        molecule_id = file_path.stem  # Gets the filename without extension
        expected_output = save_dir / f"{molecule_id}_dxtb_result.pt"
        
        if not expected_output.exists():
            args_list.append((str(file_path), molecule_id, str(save_dir), scp_mode))

    if not args_list:
        print("No files left to process.")
        return

    print(f"Running parallel calculations with {n_processes} processes...")
    print(f"Sample arguments: {args_list[0]}")
    
    # Calculate an optimal chunksize - cap it to ensure progress bar updates frequently
    chunksize = min(100, max(1, len(args_list) // (n_processes * 4)))

    # Using imap_unordered for better performance when order doesn't matter
    with multiprocessing.Pool(n_processes, initializer=worker_init) as pool:
        with tqdm(total=len(args_list)) as pbar:
            for _ in pool.imap_unordered(calculate_dxtb, args_list, chunksize=chunksize):
                pbar.update(1)
    
if __name__ == "__main__":
    main()