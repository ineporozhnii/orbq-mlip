import os
import tarfile
import torch
import io
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def process_bytes(byte_data):
    try:
        data = torch.load(io.BytesIO(byte_data), map_location='cpu', weights_only=False)
        if 'dxtb_calc_time' in data:
            return float(data['dxtb_calc_time'])
    except Exception as e:
        return None
    return None

def main():
    tar_path = "/scratch/igornep/MLGuessQC/data/OMol25_metal_neutral_dataset/OMol25_metal_neutral_clean_refcorr.tar"
    npy_path = "/scratch/igornep/MLGuessQC/mlguess/scripts/calc_times.npy"
    
    if os.path.exists(npy_path):
        print(f"Loading calculated times from {npy_path}...")
        calc_times = np.load(npy_path).tolist()
    else:
        print(f"Reading from {tar_path}...")
        
        calc_times = []
        
        # We use a multiprocessing pool with 96 workers
        pool = mp.Pool(95)
        
        def byte_generator():
            with tarfile.open(tar_path, "r|") as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith('.pt'):
                        f = tar.extractfile(member)
                        if f is not None:
                            yield f.read()

        # Pass the byte streams to the worker pool. 
        # imap_unordered will consume the generator as workers become available.
        for time_val in tqdm(pool.imap_unordered(process_bytes, byte_generator(), chunksize=100)):
            if time_val is not None:
                calc_times.append(time_val)
                
        pool.close()
        pool.join()
        
        if calc_times:
            print(f"Saving extracted times to {npy_path}...")
            np.save(npy_path, np.array(calc_times))

    if not calc_times:
        print("No calculation times found.")
        return

    calc_times = np.array(calc_times)
    
    #calc_times = calc_times[calc_times < 100]
    mean_time = np.mean(calc_times)
    median_time = np.median(calc_times)
    
    print(f"Found {len(calc_times)} calculation times. Mean: {mean_time:.3f} s, Median: {median_time:.3f} s")
    
    plt.figure(figsize=(6, 6))
    plt.hist(calc_times, bins=50, alpha=0.75, color='cornflowerblue', edgecolor='black', label=f'Mean Time: {mean_time:.3f} s\nMedian Time: {median_time:.3f} s')
    plt.xlabel('Calculation Time (s)', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.title('Histogram of dxTB Calculation Times')
    plt.legend(fontsize=18)
    plt.yscale('log')  # Use logarithmic scale for better visibility of distribution
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_file = "/scratch/igornep/MLGuessQC/mlguess/scripts/calc_time_histogram.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {out_file}")

if __name__ == "__main__":
    main()
