"""Compute GFN1-xTB orbital charges with dxtb (step 2 of the data pipeline).

For every structure ``<id>.pt`` produced by ``orbq.datagen.extract`` this runs a
GFN1-xTB single point and stores ``<id>_dxtb_result.pt`` with:

- ``dxtb_charges_orbital``: per-shell blocks of Q = N0 - P S, flattened in
  eSCN (s, p, d per atom) orbital order. The diagonal of each block holds the
  orbital charges; see Eq. (3) / Appendix B of the paper.
- ``dxtb_charges_atomic``: orbital charges summed per atom.
- ``dxtb_potential``, ``dxtb_emo``, ``dxtb_energy``, ``dxtb_n_scf_iter``,
  ``dxtb_calc_time``.

Run it once for each SCF convergence target (``--scp-mode charge|potential|fock``)
into separate output directories; ``orbq.datagen.prepare`` keeps only the
structures whose orbital charges agree across all three modes.

Usage::

    python -m orbq.datagen.dxtb_charges --input-dir extracted/ \
        --output-dir gfn1_charge/ --scp-mode charge --num-workers 32
"""

import os

# one thread per worker process: parallelism comes from multiprocessing
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing
import time
from pathlib import Path

import dxtb
import numpy as np
import torch
from tad_mctc.data.getters import get_zvalence
from tad_mctc.units import AA2AU
from tqdm import tqdm

# dxtb SCF convergence targets (dxtb.labels.SCP_MODE_*)
SCP_MODES = {"fock": 0, "charge": 1, "potential": 2}

# Reordering of the (2l+1) components of each shell from dxtb's native real
# spherical-harmonic order into the order used by the training data of the paper.
# For p this is e3nn's (x, y, z) order; for d it is the standard z-polar real d
# set (xy, yz, 3z^2-r^2, xz, x^2-y^2).
M_ORDER = {0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3]}

# e3nn uses y as the polar axis, so its l=2 basis (xz, xy, 3y^2-r^2, yz, z^2-x^2)
# is not a permutation of the z-polar d set above: two components are rotated
# into each other. This orthogonal matrix maps z-polar d functions to e3nn's l=2
# basis, so that d blocks transform exactly as D(R) Q D(R)^T.
_S3 = 3**0.5 / 2
D_TO_E3NN = torch.tensor(
    [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, 0.0, -_S3],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, _S3, 0.0, -0.5],
    ],
    dtype=torch.double,
)

# "paper": labels as used in the paper (s, p e3nn-equivariant; d in the z-polar basis)
# "e3nn":  additionally rotate d blocks into e3nn's basis (exactly equivariant)
D_BASES = ("paper", "e3nn")


def calculate_dxtb(args: tuple):
    """
    Calculate dxTB single point energy and save results.
    
    Args:
        args: Tuple containing (pt_path, molecule_id, save_dir, scp_mode[, d_basis]),
            where d_basis is one of D_BASES (default "paper").
    
    Returns:
        float: Final calculated energy
    """
    
    pt_path, molecule_id, save_dir, scp_mode = args[:4]
    d_basis = args[4] if len(args) > 4 else "paper"
    if d_basis not in D_BASES:
        raise ValueError(f"d_basis must be one of {D_BASES}, got {d_basis!r}")
    
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
    
    # Extract charge and spin multiplicity
    charge = data.get("charge", torch.tensor(0.0)).to(torch.double).squeeze()
    multiplicity = data.get("spin", torch.tensor(1.0)).to(torch.double).squeeze()
    unpaired = multiplicity - 1.0
    
    # Sanity check: Parity of electrons vs unpaired electrons
    n_electrons = torch.sum(get_zvalence(numbers)) - charge
    if int(n_electrons.item()) % 2 != int(unpaired.item()) % 2:
        return None
        
    # Ensure batched shapes for dxtb
    charge_t = charge.unsqueeze(0)
    unpaired_t = unpaired.unsqueeze(0)
    
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
        result = calc.singlepoint(positions, charge=charge_t, spin=unpaired_t)
    except Exception as e:
        print(f"Error in calculation for molecule {molecule_id}: {e}")
        return None
    calc_time = time.time() - start
    
    from dxtb._src.scf.iterator import get_refocc

    # Extract and convert results
    n_scf_iter = torch.tensor([result.iter])
    calc_time_tensor = torch.tensor([calc_time])
    final_potential = result.potential.mono.detach().cpu()
    final_emo = result.emo.detach().cpu()
    final_energy = result.total.detach().cpu()
    atomic_charges = calc.ihelp.reduce_orbital_to_atom(result.charges.mono).detach().cpu()
    
    # Equivariant Orbital Charges (Q_true)
    P_true = result.density.squeeze()
    S_true = calc.integrals.build_overlap(positions).squeeze()
    M_true = P_true @ S_true
    
    n0, occ = get_refocc(calc.integrals.hcore.refocc, charge_t, unpaired_t, calc.ihelp)
    N0_mat = torch.diag(n0.squeeze())
    Q_true = N0_mat - M_true
    
    # Verify diagonal matches orbital charges
    mono_charges = result.charges.mono.squeeze()
    assert torch.allclose(torch.diagonal(Q_true), mono_charges, atol=1e-4), "Diagonals of Q do not match orbital charges!"
    
    # Sort and extract blocks
    n_diagonals = Q_true.shape[0]
    dummy_indices = torch.arange(n_diagonals, dtype=torch.float)
    
    sort_perm = dxtb_to_escn_orbitals(
        angular_momentum_per_shell=calc.ihelp.angular,
        property_tensor=dummy_indices,
        map_shell_to_atom_idx=calc.ihelp.shells_to_atom,
        map_orbital_to_shell_idx=calc.ihelp.orbitals_to_shell
    ).long()
    
    Q_sorted = Q_true[sort_perm][:, sort_perm]

    # Reorder the m components of every shell (see M_ORDER) and optionally rotate
    # d shells into e3nn's basis (see D_TO_E3NN). Shells are contiguous after the
    # (atom, l, orbital index) sort above.
    shell_l = calc.ihelp.angular[calc.ihelp.orbitals_to_shell][sort_perm]
    m_perm = torch.arange(n_diagonals)
    basis_change = torch.eye(n_diagonals, dtype=Q_sorted.dtype)
    pos = 0
    while pos < n_diagonals:
        l = int(shell_l[pos])
        width = 2 * l + 1
        m_perm[pos : pos + width] = pos + torch.tensor(M_ORDER[l])
        if l == 2 and d_basis == "e3nn":
            basis_change[pos : pos + width, pos : pos + width] = D_TO_E3NN.to(Q_sorted.dtype)
        pos += width
    Q_sorted = Q_sorted[m_perm][:, m_perm]
    Q_sorted = basis_change @ Q_sorted @ basis_change.T

    blocks = []
    offset = 0
    # One block per (atom, angular momentum l): if an element has several shells
    # with the same l (e.g. 1s and 2s of H in GFN1), they share one block of size
    # (n_shells_l * (2l + 1))^2. This is the layout MultitaskLightningModule expects
    # (see dataset_orb_index / dataset_to_model_mask in construct_orbitals).
    sorted_lm = calc.ihelp.angular[calc.ihelp.orbitals_to_shell][sort_perm]
    sorted_atom = calc.ihelp.shells_to_atom[calc.ihelp.orbitals_to_shell][sort_perm]

    for atom_idx in range(len(numbers)):
        atom_mask = sorted_atom == atom_idx
        lms_for_atom = sorted_lm[atom_mask]

        if lms_for_atom.numel() > 0:
            _, counts = torch.unique_consecutive(lms_for_atom, return_counts=True)
            for size in counts:
                block = Q_sorted[offset:offset+size, offset:offset+size]
                blocks.append(block.flatten())
                offset += size
                
    orbital_charges = torch.cat(blocks).detach().cpu()
    
    # Update the original data dictionary with dxtb results
    data["dxtb_charges_orbital"] = orbital_charges
    data["dxtb_charges_atomic"] = atomic_charges
    data["dxtb_potential"] = final_potential
    data["dxtb_emo"] = final_emo
    data["dxtb_energy"] = final_energy
    data["dxtb_n_scf_iter"] = n_scf_iter
    data["dxtb_calc_time"] = calc_time_tensor
    data["dxtb_d_basis"] = d_basis
    
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


def _worker_init():
    torch.set_num_threads(1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute GFN1-xTB orbital/atomic charges with dxtb for extracted structures."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with <id>.pt structures (output of orbq.datagen.extract).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for <id>_dxtb_result.pt files.",
    )
    parser.add_argument(
        "--scp-mode",
        choices=sorted(SCP_MODES),
        default="charge",
        help="SCF convergence target used by dxtb (default: charge).",
    )
    parser.add_argument(
        "--d-basis",
        choices=D_BASES,
        default="paper",
        help=(
            "Basis of d-orbital blocks. 'paper' reproduces the labels used in the paper; "
            "'e3nn' rotates d blocks into e3nn's real spherical-harmonic basis so that "
            "they are exactly rotation-equivariant (see README)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes (one structure per process).",
    )
    args = parser.parse_args(argv)

    try:
        multiprocessing.set_start_method("spawn")  # safe with PyTorch
    except RuntimeError:
        pass

    if not args.input_dir.is_dir():
        parser.error(f"input directory {args.input_dir} does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scp_mode = SCP_MODES[args.scp_mode]

    # skip structures that were already computed (allows restarting)
    args_list = [
        (str(path), path.stem, str(args.output_dir), scp_mode, args.d_basis)
        for path in sorted(args.input_dir.glob("*.pt"))
        if not (args.output_dir / f"{path.stem}_dxtb_result.pt").exists()
    ]
    if not args_list:
        print("No files left to process.")
        return

    n_processes = max(1, min(args.num_workers, len(args_list)))
    print(
        f"Running {len(args_list)} dxtb calculations ({args.scp_mode} mode) "
        f"on {n_processes} processes..."
    )
    chunksize = min(100, max(1, len(args_list) // (n_processes * 4)))
    with multiprocessing.Pool(n_processes, initializer=_worker_init) as pool:
        for _ in tqdm(
            pool.imap_unordered(calculate_dxtb, args_list, chunksize=chunksize),
            total=len(args_list),
        ):
            pass


if __name__ == "__main__":
    main()
