import json
import random
from typing import Optional, Dict
from pathlib import Path
from tqdm import tqdm

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset

from mlguess.common.pyg import DataLoader

def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)

class DataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        dataset: DictConfig,
        split_file: Optional[Path],
        num_workers: DictConfig,
        batch_size: DictConfig,
        metadata_file: Optional[str] = None,
    ):
        super().__init__()
        self.dataset = dataset
        with open(split_file, "r") as fp:
            self.splits = json.load(fp)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.metadata_file = metadata_file
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.metadata: Optional[Dict] = None
        
    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        self.dataset = hydra.utils.instantiate(self.dataset)
        self.train_dataset = Subset(self.dataset, self.splits["train"])
        self.val_dataset = Subset(self.dataset, self.splits["validation"])
        self.test_dataset = Subset(self.dataset, self.splits["test"])
        
        metadata_path = Path(self.metadata_file) if self.metadata_file is not None else Path(self.dataset.path) / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as fp:
                self.metadata = json.load(fp)
        else:
            self.metadata = self.get_metadata(metadata_path)

    def get_metadata(self, metadata_path):
        energy_sum = 0
        forces_sum = 0
        orbital_charges_dxtb_sum = 0
        atomic_charges_dxtb_sum = 0
        atomic_charges_lowdin_sum = 0
        atomic_charges_mulliken_sum = 0
        energy_2 = 0
        forces_2 = 0
        orbital_charges_dxtb_2 = 0
        atomic_charges_dxtb_2 = 0
        atomic_charges_lowdin_2 = 0
        atomic_charges_mulliken_2 = 0
        
        unique_atom_types = set()
        avg_num_neighbors = 0
        progress = tqdm(total=len(self.train_dataset))
        for data in self.train_dataset:
            energy_sum += data.energy.mean()
            if hasattr(data, 'forces') and data.forces is not None:
                forces_sum += data.forces.mean()
                forces_2 += (data.forces ** 2).mean()
            atomic_charges_dxtb_sum += data.atomic_charges_dxtb.mean()
            atomic_charges_lowdin_sum += data.atomic_charges_lowdin.mean()
            atomic_charges_mulliken_sum += data.atomic_charges_mulliken.mean()
            orbital_charges_dxtb_sum += data.orbital_charges_dxtb.mean()
            energy_2 += (data.energy ** 2).mean()
            atomic_charges_dxtb_2 += (data.atomic_charges_dxtb ** 2).mean()
            atomic_charges_lowdin_2 += (data.atomic_charges_lowdin ** 2).mean()
            atomic_charges_mulliken_2 += (data.atomic_charges_mulliken ** 2).mean()
            orbital_charges_dxtb_2 += (data.orbital_charges_dxtb ** 2).mean()
            
            unique_atom_types.update(data.atom_types.numpy().tolist())
            avg_num_neighbors += data.edge_index.shape[1] / len(data.atom_types) / 2
            progress.update(1)

        energy_mean = energy_sum / len(self.train_dataset)
        energy_var = energy_2 / len(self.train_dataset) - energy_mean ** 2
        
        if hasattr(self.train_dataset[0], 'forces') and self.train_dataset[0].forces is not None:
            forces_mean = forces_sum / len(self.train_dataset)
            forces_var = forces_2 / len(self.train_dataset) - forces_mean ** 2
        else:
            import torch
            forces_mean = torch.tensor(0.0)
            forces_var = torch.tensor(1.0)
        orbital_charges_dxtb_mean = orbital_charges_dxtb_sum / len(self.train_dataset)
        orbital_charges_dxtb_var = orbital_charges_dxtb_2 / len(self.train_dataset) - orbital_charges_dxtb_mean ** 2
        atomic_charges_dxtb_mean = atomic_charges_dxtb_sum / len(self.train_dataset)
        atomic_charges_dxtb_var = atomic_charges_dxtb_2 / len(self.train_dataset) - atomic_charges_dxtb_mean ** 2
        atomic_charges_lowdin_mean = atomic_charges_lowdin_sum / len(self.train_dataset)
        atomic_charges_lowdin_var   = atomic_charges_lowdin_2 / len(self.train_dataset) - atomic_charges_lowdin_mean ** 2
        atomic_charges_mulliken_mean = atomic_charges_mulliken_sum / len(self.train_dataset)
        atomic_charges_mulliken_var = atomic_charges_mulliken_2 / len(self.train_dataset) - atomic_charges_mulliken_mean ** 2
        avg_num_neighbors = int(avg_num_neighbors / len(self.train_dataset))
        # this is the avg num neighbors without the probes
        metadata = {
            'energy_mean': energy_mean.item(),
            'energy_var': energy_var.item(),
            'forces_mean': forces_mean.item(),
            'forces_var': forces_var.item(),
            'orbital_charges_dxtb_mean': orbital_charges_dxtb_mean.item(),
            'orbital_charges_dxtb_var': orbital_charges_dxtb_var.item(),
            'atomic_charges_dxtb_mean': atomic_charges_dxtb_mean.item(),
            'atomic_charges_dxtb_var': atomic_charges_dxtb_var.item(),
            'atomic_charges_lowdin_mean': atomic_charges_lowdin_mean.item(),
            'atomic_charges_lowdin_var': atomic_charges_lowdin_var.item(),
            'atomic_charges_mulliken_mean': atomic_charges_mulliken_mean.item(),
            'atomic_charges_mulliken_var': atomic_charges_mulliken_var.item(),
            'avg_num_neighbors': avg_num_neighbors,
            'unique_atom_types': list(unique_atom_types)
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as fp:
            json.dump(metadata, fp)
        return metadata

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )