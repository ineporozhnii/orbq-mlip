from ase_calc.calculator import DXTB
from ase.atoms import Atoms
from ase.optimize.bfgs import BFGS
from ase.optimize.lbfgs import LBFGS
from ase.md.verlet import VelocityVerlet
from ase.units import fs
from pytest import approx
import numpy as np
from dxtb import IndexHelper
from dxtb import GFN1_XTB as par
from dxtb._src.scf.guess import spread_charges_atomic_to_orbital
import torch

def calculate_atomic_charges(atomic_numbers, orbital_charges):
    """
    Calculates total atomic charges by summing orbital contributions from a flat array.

    Uses a dictionary mapping atomic numbers to the number of orbitals/contributions
    per atom type to partition the flat orbital_charges array.

    Args:
        atomic_numbers (list, tuple, or np.ndarray): A sequence of integers
        representing the atomic number for each atom in the system.
        orbital_charges (np.ndarray): A 1D NumPy array containing the charges
        associated with each orbital (or contribution), concatenated for all
        atoms in the order specified by atomic_numbers.

    Returns:
        np.ndarray: A 1D NumPy array containing the total calculated atomic charge
                    for each atom corresponding to the atomic_numbers input.

    Raises:
        TypeError: If inputs have incorrect types (e.g., orbital_charges not a
                NumPy array, orbitals_per_atom_type not a dict, wrong types
                within inputs).
        ValueError: If atomic_numbers is empty, if orbital_charges is not 1D,
                    if an atomic number in atomic_numbers is not found as a key
                    in orbitals_per_atom_type, if the total number of expected
                    orbital charges does not match the length of the
                    orbital_charges array, or if dictionary values are not
                    non-negative integers.
    """
        
    # --- Input Validation ---
    if not isinstance(atomic_numbers, (list, tuple, np.ndarray)):
        raise TypeError("Input 'atomic_numbers' must be a list, tuple, or NumPy array.")
    if not isinstance(orbital_charges, np.ndarray):
        raise TypeError("Input 'orbital_charges' must be a NumPy array.")
    
    orbitals_per_atom_type = {1: 2, 6: 4, 7: 4, 8: 4}


    # Ensure atomic_numbers is a NumPy array for easier handling if needed
    atomic_numbers = np.array(atomic_numbers, dtype=int) # Ensure integer type

    if atomic_numbers.ndim != 1 or atomic_numbers.size == 0:
        raise ValueError("'atomic_numbers' cannot be empty or multi-dimensional.")
    if orbital_charges.ndim != 1:
        raise ValueError("'orbital_charges' must be a 1-dimensional NumPy array.")

    # Validate dictionary keys and values
    expected_total_orbitals = 0
    for i, atom_z in enumerate(atomic_numbers):
        if not isinstance(atom_z.item(), int) or atom_z <= 0:
            raise ValueError(f"Atomic number at index {i} must be a positive integer, but got {atom_z}")
        if atom_z.item() not in orbitals_per_atom_type:
            raise ValueError(
                f"Atomic number {atom_z} (at index {i}) not found as a key "
                f"in 'orbitals_per_atom_type' dictionary: {list(orbitals_per_atom_type.keys())}"
            )
        num_orbitals = orbitals_per_atom_type[atom_z.item()]
        if not isinstance(num_orbitals, int) or num_orbitals < 0:
            raise ValueError(
                f"Value for atomic number {atom_z} in 'orbitals_per_atom_type' "
                f"must be a non-negative integer, but got {num_orbitals}"
            )
        expected_total_orbitals += num_orbitals

    # Validate the total number of orbital charges provided
    if expected_total_orbitals != orbital_charges.size:
        raise ValueError(
            f"Mismatch in orbital counts: 'atomic_numbers' and "
            f"'orbitals_per_atom_type' imply {expected_total_orbitals} total "
            f"orbital charges, but 'orbital_charges' array has size {orbital_charges.size}."
        )

    # Validate orbital charge types (optional but good practice)
    if not np.issubdtype(orbital_charges.dtype, np.number):
        raise TypeError(f"Elements in 'orbital_charges' must be numeric, but got dtype {orbital_charges.dtype}")


    # --- Calculation ---
    atomic_charges_output = []
    current_orbital_index = 0

    for atom_z in atomic_numbers:
        # Look up the number of orbitals for this specific atom
        num_orbitals = orbitals_per_atom_type[atom_z.item()] # Use .item() to get Python int

        # Define the slice for this atom's orbitals in the flat array
        start_index = current_orbital_index
        end_index = current_orbital_index + num_orbitals

        # Extract the charges for this atom's orbitals
        charges_for_this_atom = orbital_charges[start_index:end_index]

        # Sum the charges for this atom
        total_charge = np.sum(charges_for_this_atom)
        atomic_charges_output.append(total_charge)

        # Update the index pointer for the next atom
        current_orbital_index = end_index

    # Return as a NumPy array
    return np.array(atomic_charges_output)


def test_gfn1xtb_bfgs():
    """Perform geometry optimization with GFN1-xTB and BFGS"""

    thr = 1.0e-3

    atoms = Atoms(
        symbols = "NHCHC2H3OC2H3ONHCH3",
        positions = np.array([
            [ 1.40704587284727, -1.26605342016611, -1.93713466561923],
            [ 1.85007200612454, -0.46824072777417, -1.50918242392545],
            [-0.03362432532150, -1.39269245193812, -1.74003582081606],
            [-0.56857009928108, -1.01764444489068, -2.61263467107342],
            [-0.44096297340282, -2.84337808903410, -1.48899734014499],
            [-0.47991761226058, -0.55230954385212, -0.55520222968656],
            [-1.51566045903090, -2.89187354810876, -1.32273881320610],
            [-0.18116520746778, -3.45187805987944, -2.34920431470368],
            [ 0.06989722340461, -3.23298998903001, -0.60872832703814],
            [-1.56668253918793,  0.00552120970194, -0.52884675001441],
            [ 1.99245341064342, -1.73097165236442, -3.08869239114486],
            [ 3.42884244212567, -1.30660069291348, -3.28712665743189],
            [ 3.87721962540768, -0.88843123009431, -2.38921453037869],
            [ 3.46548545761151, -0.56495308290988, -4.08311788302584],
            [ 4.00253374168514, -2.16970938132208, -3.61210068365649],
            [ 1.40187968630565, -2.43826111827818, -3.89034127398078],
            [ 0.40869198386066, -0.49101709352090,  0.47992424955574],
            [ 1.15591901335007, -1.16524842262351,  0.48740266650199],
            [ 0.00723492494701,  0.11692276177442,  1.73426297572793],
            [ 0.88822128447468,  0.28499001838229,  2.34645658013686],
            [-0.47231557768357,  1.06737634000561,  1.52286682546986],
            [-0.70199987915174, -0.50485938116399,  2.28058247845421],
        ]),
    )

    atoms.calc = DXTB(method="GFN1-xTB", accuracy=2.0, cache_api=False)
    opt = BFGS(atoms)
    opt.run(fmax=0.1)

    assert approx(atoms.get_potential_energy(), thr) == -951.9006674709672
    assert approx(np.linalg.norm(atoms.get_forces(), ord=2), thr) == 0.2052117803208497


def test_gfn1_xtb_0d():
    """Test ASE interface to GFN1-xTB"""
    thr = 1.0e-3

    atoms = Atoms(
        symbols = 'CHOCH2CH4CH2OH',
        positions = np.array([
            [ 1.578385,  0.147690,  0.343809],
            [ 1.394750,  0.012968,  1.413545],
            [ 1.359929, -1.086203, -0.359782],
            [ 0.653845,  1.215099, -0.221322],
            [ 1.057827,  2.180283,  0.093924],
            [ 0.729693,  1.184864, -1.311438],
            [-0.817334,  1.152127,  0.208156],
            [-1.303525,  2.065738, -0.145828],
            [-0.883765,  1.159762,  1.299260],
            [ 1.984120, -1.734446, -0.021385],
            [ 2.616286,  0.458948,  0.206544],
            [-1.627725, -0.034052, -0.311301],
            [-2.684229,  0.151015, -0.118566],
            [-1.501868, -0.118146, -1.397506],
            [-1.324262, -1.260154,  0.333377],
            [-0.417651, -1.475314,  0.076637],
        ]),
    )
    forces = np.array([
        [-0.37070590, -0.51067739, -0.27981764],
        [-0.04339461, -0.09290876,  0.22940156],
        [ 0.11141234,  0.46678720,  0.24552625],
        [ 0.04255709,  0.19019316, -0.23531997],
        [-0.01897377,  0.10810803,  0.05314982],
        [-0.07150720, -0.05182148, -0.08413638],
        [ 0.06631826,  0.10587709,  0.29833479],
        [-0.01062355,  0.02301460, -0.04964730],
        [ 0.06610108, -0.02724994,  0.09234280],
        [ 0.06519070, -0.19311773, -0.01152205],
        [ 0.23879786,  0.09871398, -0.04009526],
        [-0.04381577, -0.49997745,  0.08672818],
        [-0.23259608,  0.13735636,  0.06783414],
        [ 0.08297636, -0.09566973, -0.20602954],
        [-0.23686052,  0.57454371, -0.17194215],
        [ 0.35512370, -0.23317164,  0.00519275]
    ])
    charges = np.array([
        0.19494678,  0.01759972, -0.57108503, -0.05371086,  0.02458495,
        0.03915074, -0.06889977,  0.02521441,  0.03350058,  0.34380081,
        0.01442221,  0.20053193,  0.02454679,  0.01011690, -0.58349877,
        0.34877861,
    ])
    dipole_moment = np.array([0.76943477, 0.33021928, 0.05670150])

    atoms.calc = DXTB(method="GFN1-xTB")
    assert approx(atoms.get_potential_energy(), abs=thr) == -632.7363734598027
    assert approx(atoms.get_forces(), abs=thr) == forces
    atomic_charges = calculate_atomic_charges(atoms.numbers, atoms.get_charges())
    assert approx(atomic_charges, abs=thr) == charges