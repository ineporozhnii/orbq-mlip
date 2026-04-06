import time
import numpy as np
import torch
import dxtb
from ase.io import read
from tad_mctc.units import AA2AU


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


def process_predictions_file(file_path):
    """
    Process a predictions NPZ file to extract atomic numbers, positions, and orbital properties.
    """
    data = np.load(file_path)
    orbital_map = np.load("dxtb_experiments/dxtb_orbital_map.npz")
    
    all_numbers = []
    all_positions = []
    all_orbital_prop = []
    n_atom = data["n_atom"]
    flat_numbers = data["atom_types"]
    flat_positions = data["coords"]
    flat_orbital_prop = data["predictions"]
    start_atom_idx = 0
    start_orbital_idx = 0
    for i in range(len(n_atom)):
        n = n_atom[i]
        all_numbers.append(flat_numbers[start_atom_idx:start_atom_idx + n])
        all_positions.append(flat_positions[start_atom_idx:start_atom_idx + n])
        n_orbitals = 0
        for atom in flat_numbers[start_atom_idx:start_atom_idx + n]:
            n_orbitals += orbital_map[str(atom)]
 
        all_orbital_prop.append(flat_orbital_prop[start_orbital_idx:start_orbital_idx + n_orbitals])
        start_atom_idx += n
        start_orbital_idx += n_orbitals
    
    return all_numbers, all_positions, all_orbital_prop


def calculate_dxtb(args: tuple):
    """
    Calculate dxTB single point energy and save results.
    
    Args:
        args: Tuple containing (pt_path, molecule_id, save_dir, scp_mode)
    
    Returns:
        float: Final calculated energy
    """
    
    pt_path, molecule_id, save_dir, scp_mode = args
    
    # Load pt file
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except EOFError:
        print(f"Error (corrupted file): Could not load {pt_path}")
        return None
        
    # Extract atomic numbers and coordinates
    numbers = data["atom_types"].detach().to(torch.int64)
    positions = data["coords"].detach().to(torch.double) * AA2AU  # Convert Angstroms to atomic units
    positions.requires_grad_(True)
    
    # Set up calculator configuration
    calc_config = {
        "dtype": torch.double,
        "device": torch.device("cpu"),
        "opts": {"verbosity": 0, "maxiter": 250, 'scp_mode': scp_mode, "cache_charges": False, "cache_iterations": False}
    }
    
    # Initialize GFN1 calculator
    calc = dxtb.calculators.GFN1Calculator(numbers, **calc_config)
    
    # Measure calculation time
    start = time.time()
    try:
        result = calc.singlepoint(positions)
    except Exception as e:
        print(f"Error in calculation for molecule {molecule_id}: {e}")
        return None
    calc_time = time.time() - start
    
    # Extract and convert results
    n_scf_iter = torch.tensor([result.iter])
    calc_time_tensor = torch.tensor([calc_time])
    final_potential = result.potential.mono.detach().cpu()
    final_emo = result.emo.detach().cpu()
    final_energy = result.total.detach().cpu()
    orbital_charges = result.charges.mono.detach().cpu()
    atomic_charges = calc.ihelp.reduce_orbital_to_atom(result.charges.mono).detach().cpu()
    
    # Update the original data dictionary with dxtb results
    data["dxtb_charges_orbital"] = orbital_charges
    data["dxtb_charges_atomic"] = atomic_charges
    data["dxtb_potential"] = final_potential
    data["dxtb_emo"] = final_emo
    data["dxtb_energy"] = final_energy
    data["dxtb_n_scf_iter"] = n_scf_iter
    data["dxtb_calc_time"] = calc_time_tensor
    
    # Save all results to a pytorch file
    torch.save(data, f"{save_dir}/{molecule_id}_dxtb_result.pt")
    
    return final_energy.sum().item()



def dxtb_to_escn_orbitals(
    angular_momentum_per_shell: torch.Tensor,
    property_tensor: torch.Tensor,
    map_shell_to_atom_idx: torch.Tensor,
    map_orbital_to_shell_idx: torch.Tensor
) -> torch.Tensor:
    """
    Re-sorts orbitally-resolved properties to follow s, p, d ordering per atom,
    using efficient tensorized operations.

    The sorting is done as follows:
    1. For each orbital, its shell and associated atom are identified. The angular
       momentum (lm) for the orbital is taken from 'angular_momentum_per_shell'.
       These are derived using vectorized tensor indexing.
    2. All orbitals are then sorted globally based on a lexicographical key:
       a. Primary sort key: Atom index.
       b. Secondary sort key: Angular momentum (0=s, 1=p, 2=d).
       c. Tertiary sort key: Original orbital index (to ensure stability for
          orbitals of the same atom and lm type).
    3. The sorted permutation is applied to the 'property_tensor'.

    Args:
        angular_momentum_per_shell (torch.Tensor): A 1D tensor where each
            element is the angular momentum (0=s, 1=p, 2=d) of the
            corresponding SHELL. Its length is N_shells.
        property_tensor (torch.Tensor): A 1D tensor of the properties to be
            sorted (e.g., orbital charges). Its length is N_orbitals.
        map_shell_to_atom_idx (torch.Tensor): A 1D tensor mapping a shell index
            to an atom index. Its length must be N_shells.
        map_orbital_to_shell_idx (torch.Tensor): A 1D tensor mapping an orbital
            index to a shell index. Its values should be valid indices for
            'angular_momentum_per_shell' and 'map_shell_to_atom_idx'.
            Its length must be N_orbitals.

    Returns:
        torch.Tensor: A new tensor with the properties re-sorted according to
                      the s, p, d convention for each atom.
    """

    # --- Input Validation ---
    if not all(isinstance(t, torch.Tensor) for t in [angular_momentum_per_shell, property_tensor, map_shell_to_atom_idx, map_orbital_to_shell_idx]):
        raise TypeError("All inputs must be PyTorch tensors.")

    if not (angular_momentum_per_shell.dim() == 1 and \
            property_tensor.dim() == 1 and \
            map_shell_to_atom_idx.dim() == 1 and \
            map_orbital_to_shell_idx.dim() == 1):
        raise ValueError("All input tensors must be 1-dimensional.")

    num_orbitals = property_tensor.shape[0]
    num_shells = angular_momentum_per_shell.shape[0]

    if num_orbitals == 0:
        return torch.tensor([], dtype=property_tensor.dtype, device=property_tensor.device)

    if num_shells == 0:
        raise ValueError("angular_momentum_per_shell cannot be empty if property_tensor is not empty.")

    if map_orbital_to_shell_idx.shape[0] != num_orbitals:
        raise ValueError(f"map_orbital_to_shell_idx (len {map_orbital_to_shell_idx.shape[0]}) "
                         f"must have the same length as property_tensor (len {num_orbitals}).")

    if map_shell_to_atom_idx.shape[0] != num_shells:
        raise ValueError(f"map_shell_to_atom_idx (len {map_shell_to_atom_idx.shape[0]}) "
                         f"must have the same length as angular_momentum_per_shell (len {num_shells}).")

    if map_orbital_to_shell_idx.numel() > 0:
        max_shell_idx_mapped = map_orbital_to_shell_idx.max().item()
        min_shell_idx_mapped = map_orbital_to_shell_idx.min().item()
        if max_shell_idx_mapped >= num_shells:
            raise ValueError(f"Max value in map_orbital_to_shell_idx ({max_shell_idx_mapped}) "
                             f"is out of bounds for shells (0 to {num_shells-1}). Shells are indexed by angular_momentum_per_shell.")
        if min_shell_idx_mapped < 0:
            raise ValueError(f"Min value in map_orbital_to_shell_idx ({min_shell_idx_mapped}) cannot be negative.")
    elif num_orbitals > 0:
        raise ValueError("map_orbital_to_shell_idx is empty but there are orbitals.")

    # --- Determine target device for operations ---
    device = property_tensor.device

    # --- Prepare sorting keys as tensors on the target device ---
    original_indices = torch.arange(num_orbitals, device=device)

    # Ensure mapping tensors and data tensors are on the correct device for indexing/operations
    _map_orbital_to_shell_idx = map_orbital_to_shell_idx.to(device, non_blocking=True)
    _angular_momentum_per_shell = angular_momentum_per_shell.to(device, non_blocking=True)
    _map_shell_to_atom_idx = map_shell_to_atom_idx.to(device, non_blocking=True)
    _property_tensor = property_tensor # Already on 'device' by definition

    # Calculate per-orbital atom indices and LMs using vectorized indexing
    # shell_indices_of_orbitals is _map_orbital_to_shell_idx
    atom_idx_for_orbitals = _map_shell_to_atom_idx[_map_orbital_to_shell_idx]
    lm_for_orbitals = _angular_momentum_per_shell[_map_orbital_to_shell_idx]

    # --- Perform Lexicographical Sort ---
    # torch.lexsort sorts by rows (keys), from last to first.
    # Primary sort key: atom_idx_for_orbitals
    # Secondary sort key: lm_for_orbitals
    # Tertiary sort key (for stability): original_indices
    # So, keys tuple should be (tertiary, secondary, primary)
    
    sorted_permutation = np.lexsort(
        keys=(original_indices, lm_for_orbitals, atom_idx_for_orbitals)
    )
    
    # --- Apply Permutation ---
    sorted_property_tensor = _property_tensor[sorted_permutation]
            
    return sorted_property_tensor


def escn_to_dxtb_orbitals(
    sorted_property_tensor: torch.Tensor,
    angular_momentum_per_shell: torch.Tensor,
    map_shell_to_atom_idx: torch.Tensor,
    map_orbital_to_shell_idx: torch.Tensor
) -> torch.Tensor:
    """
    Reverts a sorted property tensor back to its original order before sorting
    by the `sort_orbital_properties_by_atom_and_lm_efficient` function.

    This function is the inverse of `sort_orbital_properties_by_atom_and_lm_efficient`.
    It uses the same mapping tensors to re-calculate the sorting permutation
    that was originally applied, and then inverts this permutation to restore
    the original order of properties.

    Args:
        sorted_property_tensor (torch.Tensor): A 1D tensor of properties that
            has already been sorted by the
            `sort_orbital_properties_by_atom_and_lm_efficient` function.
            Its length is N_orbitals.
        angular_momentum_per_shell (torch.Tensor): A 1D tensor where each
            element is the angular momentum (0=s, 1=p, 2=d) of the
            corresponding SHELL. Its length is N_shells. (Same as used in sorting)
        map_shell_to_atom_idx (torch.Tensor): A 1D tensor mapping a shell index
            to an atom index. Its length must be N_shells. (Same as used in sorting)
        map_orbital_to_shell_idx (torch.Tensor): A 1D tensor mapping an orbital
            index to a shell index. Its values should be valid indices for
            'angular_momentum_per_shell' and 'map_shell_to_atom_idx'.
            Its length must be N_orbitals. (Same as used in sorting)

    Returns:
        torch.Tensor: A new tensor with the properties in their original,
                      unsorted order.
    """

    # --- Input Validation ---
    if not all(isinstance(t, torch.Tensor) for t in [sorted_property_tensor, angular_momentum_per_shell, map_shell_to_atom_idx, map_orbital_to_shell_idx]):
        raise TypeError("All inputs must be PyTorch tensors.")

    if not (sorted_property_tensor.dim() == 1 and \
            angular_momentum_per_shell.dim() == 1 and \
            map_shell_to_atom_idx.dim() == 1 and \
            map_orbital_to_shell_idx.dim() == 1):
        raise ValueError("All input tensors must be 1-dimensional.")

    num_orbitals = sorted_property_tensor.shape[0]
    num_shells = angular_momentum_per_shell.shape[0]

    if num_orbitals == 0:
        return torch.tensor([], dtype=sorted_property_tensor.dtype, device=sorted_property_tensor.device)

    if num_shells == 0: # Must have shells if there are orbitals
        raise ValueError("angular_momentum_per_shell cannot be empty if sorted_property_tensor is not empty.")

    if map_orbital_to_shell_idx.shape[0] != num_orbitals:
        raise ValueError(f"map_orbital_to_shell_idx (len {map_orbital_to_shell_idx.shape[0]}) "
                         f"must have the same length as sorted_property_tensor (len {num_orbitals}).")

    if map_shell_to_atom_idx.shape[0] != num_shells:
        raise ValueError(f"map_shell_to_atom_idx (len {map_shell_to_atom_idx.shape[0]}) "
                         f"must have the same length as angular_momentum_per_shell (len {num_shells}).")

    if map_orbital_to_shell_idx.numel() > 0:
        max_shell_idx_mapped = map_orbital_to_shell_idx.max().item()
        min_shell_idx_mapped = map_orbital_to_shell_idx.min().item()
        if max_shell_idx_mapped >= num_shells:
            raise ValueError(f"Max value in map_orbital_to_shell_idx ({max_shell_idx_mapped}) "
                             f"is out of bounds for shells (0 to {num_shells-1}). Shells are indexed by angular_momentum_per_shell.")
        if min_shell_idx_mapped < 0:
            raise ValueError(f"Min value in map_orbital_to_shell_idx ({min_shell_idx_mapped}) cannot be negative.")
    elif num_orbitals > 0:
        raise ValueError("map_orbital_to_shell_idx is empty but there are orbitals.")

    # --- Determine target device for operations ---
    device = sorted_property_tensor.device

    # --- Re-calculate the sorting permutation (exactly as in the forward function) ---
    original_indices = torch.arange(num_orbitals, device=device)

    _map_orbital_to_shell_idx = map_orbital_to_shell_idx.to(device, non_blocking=True)
    _angular_momentum_per_shell = angular_momentum_per_shell.to(device, non_blocking=True)
    _map_shell_to_atom_idx = map_shell_to_atom_idx.to(device, non_blocking=True)
    
    atom_idx_for_orbitals = _map_shell_to_atom_idx[_map_orbital_to_shell_idx]
    lm_for_orbitals = _angular_momentum_per_shell[_map_orbital_to_shell_idx]
    
    # sorting_permutation[new_idx] = old_idx
    # This is the permutation that, when applied to the original unsorted tensor, yields the sorted tensor.
    sorting_permutation = np.lexsort(
        keys=(original_indices, lm_for_orbitals, atom_idx_for_orbitals)
    )
    
    # --- Apply inverse permutation to restore original order ---
    # If P_sorted = P_original[sorting_permutation],
    # then P_original[sorting_permutation[i]] = P_sorted[i] for all i.
    # This operation effectively "scatters" the values from P_sorted back to their
    # original positions in P_original.
    
    unsorted_property_tensor = torch.empty_like(sorted_property_tensor, device=device) # Ensure correct device
    unsorted_property_tensor[sorting_permutation] = sorted_property_tensor.to(device, non_blocking=True) # Ensure sorted_property_tensor is on the target device

    return unsorted_property_tensor