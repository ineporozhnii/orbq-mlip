import os
import torch
import multiprocessing as mp
from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm

# Global variable to hold the dataset instance per worker process
_dataset = None


def worker_init(data_path):
    global _dataset
    _dataset = AseDBDataset({"src": data_path})


def worker_process(args):
    dataset_idx, save_path, target_data_id, target_charge, target_spin = args
    file_path = os.path.join(save_path, f"{target_data_id}_{dataset_idx:08d}.pt")
    if os.path.exists(file_path):
        return False  # Skip if file already exists
    atoms = _dataset.get_atoms(dataset_idx)

    # Filter conditions
    if atoms.info.get("data_id") != target_data_id:
        return False
    if atoms.info.get("charge") != target_charge:
        return False
    if atoms.info.get("spin") != target_spin:
        return False

    # Prepare data dict with torch tensors
    data_dict = {
        "coords": torch.tensor(atoms.positions, dtype=torch.float32),
        "atom_types": torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        "total_energy": torch.tensor(atoms.get_total_energy(), dtype=torch.float64),
        "forces": torch.tensor(atoms.get_forces(), dtype=torch.float32),
        "mulliken_charges": (
            torch.tensor(atoms.info.get("mulliken_charges", []), dtype=torch.float32)
            if "mulliken_charges" in atoms.info
            else None
        ),
        "lowdin_charges": (
            torch.tensor(atoms.info.get("lowdin_charges", []), dtype=torch.float32)
            if "lowdin_charges" in atoms.info
            else None
        ),
        "charge": torch.tensor(atoms.info["charge"], dtype=torch.float32),
        "spin": torch.tensor(atoms.info["spin"], dtype=torch.float32),
        "idx": torch.tensor(dataset_idx, dtype=torch.long),
        "OMol25_n_scf_steps": torch.tensor(
            atoms.info.get("n_scf_steps", 0), dtype=torch.long
        ),
        "OMol25_data_id": atoms.info.get("data_id", ""),
        "OMol25_source": atoms.info.get("source", ""),
    }

    # Save to pt file using the dataset index
    torch.save(data_dict, file_path)

    return True


def process_and_save_dataset(
    data_path: str,
    save_path: str,
    n_max: int = float("inf"),
    target_data_id: str = "metal_complexes",
    target_charge: int = 0,
    target_spin: int = 1,
):
    os.makedirs(save_path, exist_ok=True)

    # Load dataset temporarily to get length
    dataset = AseDBDataset({"src": data_path})
    total_structures = len(dataset)

    num_cores = min(64, mp.cpu_count())
    n_saved = 0

    # Create an iterator for the arguments
    args_iterator = (
        (i, save_path, target_data_id, target_charge, target_spin)
        for i in range(total_structures)
    )

    with mp.Pool(num_cores, initializer=worker_init, initargs=(data_path,)) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_process, args_iterator, chunksize=100),
            total=total_structures,
            desc="Processing structures",
        )
        for success in pbar:
            if success:
                n_saved += 1
                pbar.set_postfix(saved=n_saved)

            if n_saved >= n_max:
                print(
                    f"\nSaved {n_max} structures for data_id '{target_data_id}'. Stopping."
                )
                pool.terminate()  # Stop remaining jobs immediately
                break


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensuring safe initialization

    # Train set: 450k neutral metal complexes
    DATA_PATH = "../../data/OMol25/train"
    SAVE_PATH = "../../data/OMol25_metal_neutral"
    N_MAX = 200_000

    process_and_save_dataset(
        data_path=DATA_PATH,
        save_path=SAVE_PATH,
        n_max=N_MAX,
        target_data_id="metal_complexes",
    )
