import time
import torch
import dxtb
from ase.io import read


def load_structure(filename: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a structure from an XYZ file.
    """
    # Load the structure from an XYZ file
    atoms = read(filename)

    # Extract atomic numbers and positions
    numbers = torch.tensor([atom.number for atom in atoms], dtype=torch.int64)
    positions = torch.tensor(atoms.get_positions(), dtype=torch.double)

    return numbers, positions

def main():
    start = time.time()
    # Load the structure from an XYZ file
    filename = "data/9393.xyz"  # Replace with your file path
    numbers, positions = load_structure(filename)

    # Set device and dtype
    dd = {"dtype": torch.double, "device": torch.device("cpu")}

    # Instantiate a calculator
    calc = dxtb.calculators.GFN1Calculator(numbers, **dd)

    # Compute the energy
    pos = positions.clone().requires_grad_(True)
    energy = calc.get_energy(pos)
    print(f"Time taken: {time.time() - start:.2f} seconds")
    
if __name__ == "__main__":
    main()