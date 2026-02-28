import torch
import numpy as np
from mlguess.common.pyg import Data
from mlguess.data.utils import read_npz, make_graph

class AtomicNumberTable:
    def __init__(self, zs):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)
    
def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    "adpated from: https://github.com/ACEsuit/mace/blob/main/mace"
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)

def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    adpted from: https://github.com/ACEsuit/mace/blob/main/mace
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)

class AtomicData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def build_graph(
        cls,
        atom_types,
        atom_coords,
        cell,
        energy,
        orbital_charges_dxtb,
        atomic_charges_mulliken, 
        atomic_charges_lowdin, 
        atomic_charges_dxtb,
        metadata: str,
        z_table: AtomicNumberTable,
        atom_cutoff: float = 5.0,
        disable_pbc: bool = False,
        device: str = "cpu",
        max_neighbors: int = 24,
    ) -> "AtomicData":        
        n_atom = len(atom_coords)

        idx_src, idx_dst, shifts, unit_shifts = make_graph(
            cell.to(device),
            atom_coords.to(device),
            atom_coords.to(device),
            atom_cutoff,
            disable_pbc,
            disable_sc=True,
            max_neigh=max_neighbors
        )
        edge_index = torch.stack([idx_src, idx_dst], dim=0)
        
        indices = atomic_numbers_to_indices(atom_types, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )

        data_dict = dict(
            edge_index=edge_index,
            coords=atom_coords,
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell.unsqueeze(0),
            atom_types=atom_types,
            node_attrs=one_hot,
            n_atom=n_atom,
            num_nodes=n_atom,
            energy=energy,
            orbital_charges_dxtb=orbital_charges_dxtb,
            atomic_charges_mulliken=atomic_charges_mulliken,
            atomic_charges_lowdin=atomic_charges_lowdin,
            atomic_charges_dxtb=atomic_charges_dxtb,
            metadata=metadata
        )
        return cls(**data_dict)

    @classmethod
    def from_file(
        cls,
        fpath=None,
        fcontent=None,
        finfo=None,
        z_table: AtomicNumberTable = None,
        atom_cutoff: float = 4.0,
        disable_pbc: bool = False,
        max_neighbors: int = 24,
        device: str = "cpu",
    ):
        assert finfo.name.endswith((".npz")), "Only .npz files are supported."
        assert (fpath or (fcontent and finfo)) and not (fpath and (fcontent and finfo))

        data = read_npz(fcontent)
        metadata = finfo.name.split("/")[-1].split(".")[0]
    
        return cls.build_graph(
            *data,
            metadata=metadata,
            z_table=z_table,
            atom_cutoff=atom_cutoff,
            disable_pbc=disable_pbc,
            device=device,
            max_neighbors=max_neighbors
        )