import os
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import orbq  # noqa: F401  (registers the ${orbq_root:} resolver)

CONFIG_DIR = str(Path(orbq.__file__).resolve().parent / "config")
SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "omol25_sample"

SETTINGS = [
    "energy",
    "energy_forces",
    "energy_forces_charges_orbital_dxtb",
    "energy_forces_charges_atomic_dxtb",
    "energy_forces_charges_atomic_lowdin",
    "energy_forces_charges_atomic_mulliken",
]


def compose_cfg(overrides=(), resolve=True):
    os.environ.setdefault("ORBQ_DATA_DIR", "/nonexistent/orbq_data")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = compose(config_name="default", overrides=list(overrides))
    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture(scope="session")
def sample_datamodule():
    cfg = compose_cfg(["data=sample"])
    dm = hydra.utils.instantiate(cfg.data, _recursive_=False)
    dm.setup("fit")
    return dm


@pytest.fixture(scope="session")
def sample_batch(sample_datamodule):
    torch.manual_seed(0)
    return next(iter(sample_datamodule.train_dataloader()))


def build_model(setting, metadata):
    cfg = compose_cfg(["data=sample", f"model={setting}"])
    torch.manual_seed(0)
    return hydra.utils.instantiate(cfg.model, train=cfg.train, _recursive_=False, metadata=metadata)
