import os
import sys
import json
import shutil
import warnings
import multiprocessing
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def _worker_compare(args):
    fname, dir_charge, dir_potential, dir_fock, clean_dir, rtol, atol = args
    path_c = dir_charge / fname
    path_p = dir_potential / fname
    path_f = dir_fock / fname
    
    # Ensure one thread per worker to avoid PyTorch spawning excess threads
    torch.set_num_threads(1)
    
    try:
        data_c = torch.load(path_c, map_location="cpu", weights_only=False)
        data_p = torch.load(path_p, map_location="cpu", weights_only=False)
        data_f = torch.load(path_f, map_location="cpu", weights_only=False)
    except Exception as e:
        return (fname, False, None, None, f"Failed to load: {e}")
        
    if "dxtb_charges_orbital" in data_c and "dxtb_charges_orbital" in data_p and "dxtb_charges_orbital" in data_f:
        charges_c = data_c["dxtb_charges_orbital"]
        charges_p = data_p["dxtb_charges_orbital"]
        charges_f = data_f["dxtb_charges_orbital"]
        
        if charges_c.shape == charges_p.shape == charges_f.shape:
            if torch.allclose(charges_c, charges_p, rtol=rtol, atol=atol) and \
               torch.allclose(charges_c, charges_f, rtol=rtol, atol=atol):
                # Copy matching file to clean dir
                shutil.copy2(path_f, clean_dir / fname)
                return (fname, True, torch.min(charges_f).item(), torch.max(charges_f).item(), None)
        else:
            return (fname, False, None, None, "Shape mismatch")
    else:
        return (fname, False, None, None, "Missing 'dxtb_charges_orbital'")
        
    return (fname, False, None, None, None)

def compare_and_filter_structures(dir_charge: Path, dir_potential: Path, dir_fock: Path, clean_dir: Path, rtol: float = 1e-2, atol: float = 1e-2):
    """
    Compares dxtb_charges_orbital for pt files in three directories safely in parallel.
    If they match, saves the file to a clean directory.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    files_charge = set(f.name for f in dir_charge.glob("*.pt"))
    files_potential = set(f.name for f in dir_potential.glob("*.pt"))
    files_fock = set(f.name for f in dir_fock.glob("*.pt"))
    
    common_files = list(files_charge.intersection(files_potential).intersection(files_fock))
    print(f"Found {len(common_files)} common files. Comparing charges on multiple cores...")
    
    matched_files = []
    min_charges = []
    max_charges = []
    
    args_list = [(fname, dir_charge, dir_potential, dir_fock, clean_dir, rtol, atol) for fname in common_files]
    n_processes = min(190, multiprocessing.cpu_count())
    chunksize = min(1000, max(1, len(args_list) // (n_processes * 2)))

    with multiprocessing.Pool(n_processes) as pool:
        for result in tqdm(pool.imap_unordered(_worker_compare, args_list, chunksize=chunksize), total=len(args_list), desc="Filtering Structures"):
            fname, is_match, min_val, max_val, error = result
            if is_match:
                matched_files.append(fname)
                min_charges.append(min_val)
                max_charges.append(max_val)
            elif error:
                # Optionally print or log the error, skipping for speed unless debugging
                pass
                
    matched_count = len(matched_files)
    print(f"Saved {matched_count} exactly matching files to {clean_dir}")
    
    if matched_count > 0:
        print("Plotting distributions and filtering out structures with extreme charge values (> 5 sigma)...")
        min_charges = np.array(min_charges)
        max_charges = np.array(max_charges)
        
        mu_min, std_min = np.mean(min_charges), np.std(min_charges)
        mu_max, std_max = np.mean(max_charges), np.std(max_charges)
        
        # --- Create Plot ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        def compute_outliers(data_min, data_max, mu_mi, std_mi, mu_ma, std_ma, sigma):
            # Total outliers removed if this sigma is used for BOTH min and max simultaneously
            cond_min = np.abs(data_min - mu_mi) > sigma * std_mi
            cond_max = np.abs(data_max - mu_ma) > sigma * std_ma
            return np.sum(cond_min | cond_max)
            
        def compute_outliers_single(data, mu, std, sigma):
            return np.sum(np.abs(data - mu) > sigma * std)

        # Plot for Min Charges
        ax1.hist(min_charges, bins=100, log=True, color='blue', alpha=0.7)
        ax1.set_title("Min Orbital Charges Distribution")
        ax1.set_xlabel("Charge")
        ax1.set_ylabel("Count (log scale)")
        
        # Plot for Max Charges
        ax2.hist(max_charges, bins=100, log=True, color='red', alpha=0.7)
        ax2.set_title("Max Orbital Charges Distribution")
        ax2.set_xlabel("Charge")
        ax2.set_ylabel("Count (log scale)")
        
        for s, c in zip([3, 4, 5], ['red', 'orange', 'green']):
            total_removed = compute_outliers(min_charges, max_charges, mu_min, std_min, mu_max, std_max, s)
            label = f"{s} $\sigma$ (Total removed: {total_removed})"
            
            # Min subplot lines
            ax1.axvline(mu_min - s*std_min, color=c, linestyle='--', label=label)
            ax1.axvline(mu_min + s*std_min, color=c, linestyle='--')
            
            # Max subplot lines
            ax2.axvline(mu_max - s*std_max, color=c, linestyle='--', label=label)
            ax2.axvline(mu_max + s*std_max, color=c, linestyle='--')
            
        ax1.legend()
        ax2.legend()
        
        plot_path = clean_dir.parent / "charge_distributions_outliers.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved distribution plot to {plot_path}")
        # -------------------
        
        outliers_count = 0
        for i, fname in enumerate(matched_files):
            if abs(min_charges[i] - mu_min) > 5 * std_min or abs(max_charges[i] - mu_max) > 5 * std_max:
                outliers_count += 1
                (clean_dir / fname).unlink()
                
        print(f"Deleted {outliers_count} outliers. {matched_count - outliers_count} files remaining in clean_dir.")
        return matched_count - outliers_count
        
    return matched_count


def create_dataset_splits(clean_dir: Path, splits_dir: Path, train_sizes: list, n_val: int, n_test: int, seed: int = 42):
    """
    Splits the clean data files into train, val, test and saves to JSON files for each train size.
    The validation and test sets remain the same across all splits.
    Returns the split dictionary corresponding to the largest training set, and the sorted list of files.
    """
    files = sorted([f.name for f in clean_dir.glob("*.pt")])
    indices = np.arange(len(files))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    max_train = max(train_sizes)
    total_requested = max_train + n_val + n_test
    if len(files) < total_requested:
        raise ValueError(f"Not enough clean files. Requested {total_requested} (max), but only have {len(files)}.")
        
    val_indices = indices[-n_test - n_val : -n_test] if n_test > 0 else indices[-n_val:]
    test_indices = indices[-n_test:]
    
    largest_split = None
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for n_train in sorted(train_sizes):
        train_indices = indices[:n_train]
        splits = {
            "train": train_indices.tolist(),
            "validation": val_indices.tolist(),
            "test": test_indices.tolist()
        }
        
        # Save split JSON named by train size
        splits_json = splits_dir / f"datasplits_{n_train}.json"
        with open(splits_json, 'w') as f:
            json.dump(splits, f, indent=4)
            
        print(f"Splits saved to {splits_json} (Train: {n_train}, Val: {n_val}, Test: {n_test})")
        
        if n_train == max_train:
            largest_split = splits
            # Also save a default datasplits.json for the largest size
            default_json = splits_dir / "datasplits.json"
            with open(default_json, 'w') as f:
                json.dump(splits, f, indent=4)
            
    return largest_split, files


def train_linear_energy_model(clean_dir: Path, splits: dict, files: list):
    """
    Fits Linear Regression to total_energy of training set.
    Uses 'atom_types' as features (Bag of Atoms).
    """
    train_indices = splits["train"]
    train_files = [files[i] for i in train_indices]
    unique_atomic_numbers = set()
    train_data = []

    for fname in tqdm(train_files, desc="Reading training data"):
        data = torch.load(clean_dir / fname, map_location="cpu", weights_only=False)
        numbers = data["atom_types"].numpy()
        energy = data["total_energy"].item()
        
        unique_atomic_numbers.update(numbers)
        train_data.append((numbers, energy))

    sorted_atomic_numbers = sorted(list(unique_atomic_numbers))
    atom_to_index = {num: i for i, num in enumerate(sorted_atomic_numbers)}
    print(f"Unique atomic numbers found in train set: {sorted_atomic_numbers}")

    # Construct feature matrix X
    X_train = np.zeros((len(train_data), len(sorted_atomic_numbers)))
    y_train = np.zeros(len(train_data))

    for i, (numbers, energy) in enumerate(tqdm(train_data, desc="Building features")):
        counts = Counter(numbers)
        for atomic_num, count in counts.items():
            if atomic_num in atom_to_index:
                X_train[i, atom_to_index[atomic_num]] = count
        y_train[i] = energy

    # Fit Model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    print(f"Linear Regression R^2 on training data: {lin_reg.score(X_train, y_train):.4f}")
    return lin_reg, sorted_atomic_numbers, atom_to_index


def plot_predictions_vs_true(clean_dir: Path, splits: dict, files: list, lin_reg: LinearRegression,
                             sorted_atomic_numbers: list, atom_to_index: dict, plot_path: Path):
    """
    Evaluates the linear regression model on the validation and test sets,
    and saves a scatter plot of predictions vs true energies.
    """
    datasets = {"Validation": splits.get("validation", []), "Test": splits.get("test", [])}
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"Validation": "blue", "Test": "red"}
    
    for split_name, split_indices in datasets.items():
        if not split_indices:
            continue
            
        split_files = [files[i] for i in split_indices]
        y_true = []
        X_features = np.zeros((len(split_files), len(sorted_atomic_numbers)))
        
        for i, fname in enumerate(tqdm(split_files, desc=f"Evaluating {split_name} set")):
            data = torch.load(clean_dir / fname, map_location="cpu", weights_only=False)
            numbers = data["atom_types"].numpy()
            y_true.append(data["total_energy"].item())
            
            counts = Counter(numbers)
            for atomic_num, count in counts.items():
                if atomic_num in atom_to_index:
                    X_features[i, atom_to_index[atomic_num]] = count
                    
        if len(y_true) > 0:
            y_pred = lin_reg.predict(X_features)
            ax.scatter(y_true, y_pred, alpha=0.5, label=f"{split_name}", color=colors[split_name])
            
            from sklearn.metrics import root_mean_squared_error
            rmse = root_mean_squared_error(y_true, y_pred)
            print(f"{split_name} RMSE: {rmse:.4f}")
            
    # Draw identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  
        np.max([ax.get_xlim(), ax.get_ylim()]),  
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Energy")
    ax.set_ylabel("Predicted Baseline Energy")
    ax.set_title("Linear Fit Predictions vs True Energy")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved scatter plot to {plot_path}")


def _worker_correct(args):
    fname, clean_dir, refcorr_dir, lin_reg, sorted_atomic_numbers, atom_to_index = args
    # Ensure one thread per worker to avoid PyTorch spawning excess threads
    torch.set_num_threads(1)
    
    try:
        file_path = clean_dir / fname
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        
        numbers = data["atom_types"].numpy()
        total_energy = data["total_energy"].item()

        # Build feature vector
        feature_vector = np.zeros((1, len(sorted_atomic_numbers)))
        counts = Counter(numbers)
        for atomic_num, count in counts.items():
            if atomic_num in atom_to_index:
                feature_vector[0, atom_to_index[atomic_num]] = count

        # Predict baseline and correct
        baseline_energy = lin_reg.predict(feature_vector)[0]
        corrected_energy = float(total_energy - baseline_energy)

        # Append back into torch dictionary
        data['total_energy_corrected'] = torch.tensor(corrected_energy, dtype=torch.float32)

        # Save corrected structure
        torch.save(data, refcorr_dir / fname)
        
        return corrected_energy
    except Exception as e:
        return None


def save_corrected_energies(clean_dir: Path, refcorr_dir: Path, splits: dict, files: list, lin_reg: LinearRegression, 
                            sorted_atomic_numbers: list, atom_to_index: dict, coef_json: Path, hist_plot_path: Path):
    """
    Predicts baseline energy, updates file dictionary with 'total_energy_corrected' in parallel,
    and optionally saves coefficients to JSON. Then plots a histogram of the corrected energies.
    """
    refcorr_dir.mkdir(parents=True, exist_ok=True)
    all_splits_indices = splits["train"] + splits["validation"] + splits["test"]
    all_splits_files = [files[i] for i in all_splits_indices]

    # Generate coefficient dict
    coef_dict = {int(atomic_num): float(coef) for atomic_num, coef in zip(sorted_atomic_numbers, lin_reg.coef_)}
    coef_dict['intercept'] = float(lin_reg.intercept_)

    with open(coef_json, 'w') as f:
        json.dump(coef_dict, f, indent=4)
    print(f"Coefficients saved to {coef_json}")

    all_corrected_energies = []
    args_list = [(fname, clean_dir, refcorr_dir, lin_reg, sorted_atomic_numbers, atom_to_index) for fname in all_splits_files]
    n_processes = min(190, multiprocessing.cpu_count())
    chunksize = min(1000, max(1, len(args_list) // (n_processes * 2)))

    # Recalculate properties for all splits in parallel
    with multiprocessing.Pool(n_processes) as pool:
        for result in tqdm(pool.imap_unordered(_worker_correct, args_list, chunksize=chunksize), total=len(args_list), desc="Correcting energies"):
            if result is not None:
                all_corrected_energies.append(result)

    print(f"Corrected, clean datasets saved successfully in {refcorr_dir}")

    print("Plotting histogram of corrected energies...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_corrected_energies, bins=100, color='purple', alpha=0.7)
    ax.set_title("Corrected Total Energies Distribution")
    ax.set_xlabel("Corrected Energy")
    ax.set_ylabel("Count")
    
    fig.tight_layout()
    fig.savefig(hist_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved corrected energies histogram to {hist_plot_path}")


def main():
    # --- 1. Variables defining splits and paths ---
    base_dir = Path("/scratch/igornep/MLGuessQC/data")
    
    # Change these paths to your exact directories!
    dir_charge = base_dir / "OMol25_metal_neutral_gfn1_charge"
    dir_potential = base_dir / "OMol25_metal_neutral_gfn1_potential"
    dir_fock = base_dir / "OMol25_metal_neutral_gfn1_fock"
    
    # Output directories
    clean_dir = base_dir / "OMol25_metal_neutral_clean"
    refcorr_dir = base_dir / "OMol25_metal_neutral_clean_refcorr"
    
    # Logs and split files
    splits_dir = base_dir / "datasplits"
    coef_json = base_dir / "energy_coefficients.json"
    scatter_plot_path = base_dir / "energy_predictions_scatter.png"
    hist_plot_path = base_dir / "corrected_energies_histogram.png"
    
    # Adjust variables here for number of structures to include in each split step
    TRAIN_SIZES = [1000, 3000, 10000, 30000, 100000, 535183]
    N_VAL   = 10000
    N_TEST  = 50000

    # --- 2. Compare and Filter Data ---
    # if not clean_dir.exists() or len(list(clean_dir.glob("*.pt"))) == 0:
    #     print("Starting compare and filter step...")
    #     compare_and_filter_structures(dir_charge, dir_potential, dir_fock, clean_dir)
    # else:
    #     print(f"Clean directory '{clean_dir}' is already populated. Using existing files.")

    # --- 3. Create Dataset Splits ---
    print("Splitting dataset...")
    try:
        splits, all_files = create_dataset_splits(clean_dir, splits_dir, TRAIN_SIZES, N_VAL, N_TEST)
    except ValueError as e:
        print(f"Error during splitting: {e}")
        sys.exit(1)

    # --- 4. Train LR to Energy baseline ---
    print("Training Baseline Energy model...")
    lin_reg, sorted_atomic_numbers, atom_to_index = train_linear_energy_model(clean_dir, splits, all_files)
    
    print("Evaluating model on validation and test sets...")
    plot_predictions_vs_true(clean_dir, splits, all_files, lin_reg, sorted_atomic_numbers, atom_to_index, scatter_plot_path)

    # --- 5. Correct file energies and Save ---
    print("Correcting geometries' energies...")
    save_corrected_energies(clean_dir, refcorr_dir, splits, all_files, lin_reg, sorted_atomic_numbers, atom_to_index, coef_json, hist_plot_path)


if __name__ == "__main__":
    main()
