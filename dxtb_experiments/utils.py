import time
import numpy as np
import torch
import dxtb
from ase.io import read



def load_xyz(filename: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a structure from an XYZ file.
    """
    # Load the structure from an XYZ file
    atoms = read(filename, format="xyz")

    # Extract atomic numbers and positions
    numbers = torch.tensor([atom.number for atom in atoms], dtype=torch.int64)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.double)

    return numbers, positions


def work_lmdb(file_path: str, file_name: str):
    """
    Open and return an LMDB dataset.
    
    Args:
        file_path: Path to the directory containing the LMDB file
        file_name: Name of the LMDB file
    
    Returns:
        LmdbDataset: Dataset object for accessing LMDB data
    """
    import lmdb
    from ocpmodels.datasets import LmdbDataset
    
    # Create full path to the LMDB file
    full_path = file_path + file_name
    
    # Open LMDB environment
    lmdb_env = lmdb.open(file_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    
    # Configure LMDB with appropriate settings
    db = lmdb.open(
        full_path,
        map_size=1099511627776 * 2,  # 2TB map size
        subdir=False,
        meminit=False,
        map_async=True,
    )
  
    # Create and return the dataset
    dataset = LmdbDataset({"src": full_path})
    return dataset


def calculate_dxtb(
    numbers: np.ndarray, positions: np.ndarray, molecule_id: str, save_dir: str
) -> float:
    """
    Calculate dxTB single point energy and save results.
    
    Args:
        numbers: Atomic numbers of atoms in the molecule
        positions: Cartesian coordinates of atoms
        molecule_id: Unique identifier for the molecule
        save_dir: Directory where results will be saved
    
    Returns:
        float: Final calculated energy
    """
    # Clone and convert input tensors to appropriate types
    numbers = numbers.clone().detach().to(torch.int64)
    positions = positions.clone().detach().to(torch.double).requires_grad_(True)

    # Set up calculator configuration
    calc_config = {
        "dtype": torch.double,
        "device": torch.device("cpu"),
        "opts": {"verbosity": 0}
    }
    
    # Initialize GFN1 calculator
    calc = dxtb.calculators.GFN1Calculator(numbers, **calc_config)
    
    # Measure calculation time
    start = time.time()
    result = calc.singlepoint(positions)
    calc_time = time.time() - start
    
    # Extract and convert results
    n_scf_iter = np.array([result.iter])
    calc_time = np.array([calc_time])
    final_charges = result.charges.mono.detach().numpy()
    final_emo = result.emo.detach().numpy()
    final_energy = result.total.detach().numpy()
    
    # Load initial charges
    init_eeq_charges = torch.load(
        "dxtb_experiments/temp/eeq_charges.pt", weights_only=True
    ).detach().numpy()
    
    # Save all results to a compressed NumPy file
    np.savez(
        f"{save_dir}/{molecule_id}_dxtb_result",
        init_eeq_charges=init_eeq_charges,
        final_charges=final_charges,
        final_emo=final_emo,
        final_energy=final_energy,
        n_scf_iter=n_scf_iter,
        calc_time=calc_time
    )
    
    return final_energy