
import os
# Disable internal multithreading in child processes
# This forces each child process to be single-threaded, leaving the main
# multiprocessing pool in control of parallelism.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import torch
from tqdm import tqdm
import dxtb
import json
from dxtb_experiments.utils import escn_to_dxtb_orbitals
from dxtb_experiments.utils import process_predictions_file
import argparse
import os
import multiprocessing
import itertools

def compute_single_mol(args_tuple):
        scp_mode_map = {
            "CHARGE": dxtb.labels.SCP_MODE_CHARGE,
            "POTENTIAL": dxtb.labels.SCP_MODE_POTENTIAL,
            "FOCK": dxtb.labels.SCP_MODE_FOCK
        }
        scf_mode_map = {
            "FULL": dxtb.labels.SCF_MODE_FULL,
            "IMPLICIT_NON_PURE": dxtb.labels.SCF_MODE_IMPLICIT_NON_PURE
        }
        numbers, positions, ml_guess_prop, guess_type, scp_mode, scf_mode, max_iter, damp = args_tuple
        results = {}
        numbers_t = torch.tensor(numbers, dtype=torch.int64)
        positions_t = torch.tensor(positions, dtype=torch.double)
        ml_guess_prop_t = torch.tensor(ml_guess_prop, dtype=torch.double)

        if guess_type == "default":
            dd = {"dtype": torch.double, "device": torch.device("cpu"), "opts": {"verbosity": 0, "maxiter": max_iter, 'scp_mode': scp_mode_map[scp_mode], 'scf_mode': scf_mode_map[scf_mode], "damp_init": damp, "cache_charges": False, "cache_iterations": False}}
            calc = dxtb.calculators.GFN1Calculator(numbers_t, **dd)
            start = time.time()
            result = calc.singlepoint(positions_t)
            end = time.time()
            results['n_scf_iter'] = np.array([result.iter])
            results['calc_time'] = np.array([end - start])

        if guess_type == "ml":
            dd = {"dtype": torch.double, "device": torch.device("cpu"), "opts": {"verbosity": 0, "maxiter": max_iter, 'scp_mode': scp_mode_map[scp_mode], 'scf_mode': scf_mode_map[scf_mode], "damp_init": damp, "cache_charges": False, "cache_iterations": False}}
            calc = dxtb.calculators.GFN1Calculator(numbers_t, **dd)
            ml_guess_prop_dxtb = escn_to_dxtb_orbitals(
                sorted_property_tensor=ml_guess_prop_t,
                angular_momentum_per_shell=calc.ihelp.angular,
                map_shell_to_atom_idx=calc.ihelp.shells_to_atom,
                map_orbital_to_shell_idx=calc.ihelp.orbitals_to_shell
            )
            start = time.time()
            result = calc.singlepoint(positions_t, initial_charges=ml_guess_prop_dxtb)
            end = time.time()
            results['n_scf_iter'] = int(result.iter)
            results['calc_time'] = end - start
            results['numbers'] = numbers
        return results
    

def run_benchmark(args):
    
    print("Running DXTB benchmark...")
    print(f"Arguments: {args}")
    prediction_files = []
    
    if args.predictions_file:
        prediction_files.append(args.predictions_file)
    elif args.predictions_dir:
        prediction_files = [os.path.join(args.predictions_dir, f) for f in os.listdir(args.predictions_dir) if f.endswith('.npz')]
    
    for predictions_file in prediction_files:
        print(f"Processing file: {predictions_file}")
        all_numbers, all_positions, all_orbital_prop = process_predictions_file(predictions_file)

        n_scf_iter = []
        calc_time = []
        
        all_numbers = all_numbers[:500]  # Limit to 10 for testing
        all_positions = all_positions[:500]  # Limit to 10 for testing
        all_orbital_prop = all_orbital_prop[:500]  # Limit to 10 for testing

        # all_numbers = [all_numbers[i] for i in [290, 1081, 2562]]
        # all_positions = [all_positions[i] for i in [290, 1081, 2562]]
        # all_orbital_prop = [all_orbital_prop[i] for i in [290, 1081, 2562]]
        
        damp_values = [0.01, 0.025, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.5, 0.75]

        # Use only first 390 molecules for selecting optimal damp value
        damp_numbers = all_numbers[:200]
        damp_positions = all_positions[:200]
        damp_orbital_prop = all_orbital_prop[:200]
        
        if args.scp_mode is not None and args.scf_mode is not None:
            scpmode_scfmode_combinations = [(args.scp_mode, args.scf_mode)]
            print(f"Running {(scpmode_scfmode_combinations)} combinations of SCP and SCF modes...")
            
        if args.scp_mode is not None and args.scf_mode is None:
            scpmode_scfmode_combinations = [(args.scp_mode, scf_mode) for scf_mode in ["FULL", "IMPLICIT_NON_PURE"]]
            print(f"Running SCP mode {args.scp_mode} with all SCF modes: {scpmode_scfmode_combinations}")
        
        if args.scp_mode is None and args.scf_mode is not None:
            scpmode_scfmode_combinations = [(scp_mode, args.scf_mode) for scp_mode in ["CHARGE", "POTENTIAL", "FOCK"]]
            print(f"Running SCF mode {args.scf_mode} with all SCP modes: {scpmode_scfmode_combinations}")
        
        if args.scp_mode is None and args.scf_mode is None:
            scpmode_scfmode_combinations = list(itertools.product(
                ["CHARGE", "POTENTIAL", "FOCK"],
                ["FULL", "IMPLICIT_NON_PURE"]
            ))
            print(f"Running all combinations of SCP and SCF modes: {scpmode_scfmode_combinations}")
        
        for scp_mode, scf_mode in scpmode_scfmode_combinations:
            best_damp = None
            best_n_scf = float('inf')
            results_per_damp = {}
            for damp in damp_values:
                args_list = [
                    (numbers, positions, ml_guess_prop, args.guess_type, scp_mode, scf_mode, args.maxiter, damp)
                    for numbers, positions, ml_guess_prop in zip(damp_numbers, damp_positions, damp_orbital_prop)
                ]
                n_scf_list = []
                with multiprocessing.Pool(processes=args.n_processes) as pool:
                    for res in tqdm(pool.imap_unordered(compute_single_mol, args_list), total=len(args_list), desc=f"Damp={damp}"):
                        if res.get('n_scf_iter') is not None:
                            n_scf_list.append(res.get('n_scf_iter'))
                avg_n_scf = np.mean(n_scf_list) if n_scf_list else float('inf')
                print(f"Damp: {damp}, SCP mode: {scp_mode}, SCF mode: {scf_mode}, Average N SCF iterations: {avg_n_scf:.2f}")
                results_per_damp[damp] = n_scf_list
                if avg_n_scf < best_n_scf:
                    best_n_scf = avg_n_scf
                    best_damp = damp

            print(f"Best damp: {best_damp} with N SCF: {best_n_scf}")
            if args.save_dir:
                with open(os.path.join(args.save_dir, os.path.basename(predictions_file).replace('.npz', "_damp_benchmark_results.json")), "w") as f:
                    json.dump(results_per_damp, f, indent=2)
            
            args_list = [
                (numbers, positions, ml_guess_prop, args.guess_type, scp_mode, scf_mode, args.maxiter, damp)
                for numbers, positions, ml_guess_prop in zip(all_numbers, all_positions, all_orbital_prop)
            ]

            with multiprocessing.Pool(processes=args.n_processes) as pool:
                results = list(tqdm(pool.imap(compute_single_mol, args_list), total=len(args_list)))
            numbers = []
            for res in results:
                n_scf_iter.append(res.get('n_scf_iter'))
                calc_time.append(res.get('calc_time'))
                numbers.append(res.get('numbers'))
            numbers = np.concatenate(numbers)
            assert np.all(np.concatenate(all_numbers) == numbers), "Numbers mismatch after processing all molecules."
        
            if args.guess_type == "default":
                print(f"Default guess: Average N SCF iterations: {np.mean(n_scf_iter):.2f}, Average time: {np.mean(calc_time):.2f} seconds")
                filename = os.path.basename(predictions_file).replace('.npz', f'_{scp_mode}_{scf_mode}_DEFAULT_benchmark_results.npz')    
                
            if args.guess_type == "ml":
                print(f"ML guess: Average N SCF iterations: {np.mean(n_scf_iter):.2f}, Average time: {np.mean(calc_time):.2f} seconds")
                filename = os.path.basename(predictions_file).replace('.npz', f'_{scp_mode}_{scf_mode}_ML_benchmark_results.npz')
            
            np.savez(os.path.join(args.save_dir, filename),
                    n_scf_iter=np.array(n_scf_iter),
                    calc_time=np.array(calc_time))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DXTb benchmark.")
    parser.add_argument("--predictions_file", type=str, default=None, help="File with model predictions.")
    parser.add_argument("--predictions_dir", type=str, default=None, help="Directory with multiple model prediction files.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory for saving results.")
    parser.add_argument("--guess_type", type=str, default=None, choices=["default", "ml"], help="Select guess for benchmark.")
    parser.add_argument("--scp_mode", type=str, default=None, choices=["CHARGE", "POTENTIAL", "FOCK"], help="SCP mode for calculation.")
    parser.add_argument("--scf_mode", type=str, default=None, choices=["FULL", "IMPLICIT_NON_PURE"], help="SCF mode for calculation.")
    parser.add_argument("--n_processes", type=int, default=multiprocessing.cpu_count(), help="Number of processes for parallel execution.")
    parser.add_argument("--maxiter", type=int, default=500, help="Maximum SCF iterations.")
    args = parser.parse_args()
    
    assert (args.predictions_file is not None) != (args.predictions_dir is not None), "Specify either --predictions_file or --predictions_dir, not both."
    assert args.save_dir is not None, "Please specify --save_dir to save results."
    assert os.path.exists(args.save_dir), f"Save directory {args.save_dir} does not exist."
    assert os.path.exists(args.predictions_file) if args.predictions_file else os.path.exists(args.predictions_dir), "Predictions file or directory does not exist."
    assert args.maxiter > 0, "maxiter must be positive."
    assert args.guess_type in ["default", "ml"], "guess_type must be either 'default' or 'ml'."

    run_benchmark(args)