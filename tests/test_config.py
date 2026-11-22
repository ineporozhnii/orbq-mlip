"""Hydra configs compose and carry the paper's hyperparameters."""

import pytest
from conftest import SETTINGS, compose_cfg

# loss weights (energy, charges, forces) per training-set size, as in the paper
CHARGE_WEIGHTS = {
    "energy_forces_charges_orbital_dxtb": {3000: 9.5, 10000: 10.9, 30000: 11.9, 100000: 10.3},
    "energy_forces_charges_atomic_dxtb": dict.fromkeys((3000, 10000, 30000, 100000), 5.8),
    "energy_forces_charges_atomic_lowdin": dict.fromkeys((3000, 10000, 30000, 100000), 6.7),
    "energy_forces_charges_atomic_mulliken": dict.fromkeys((3000, 10000, 30000, 100000), 0.9),
}
FORCE_WEIGHTS = {3000: 1.6, 10000: 1.8, 30000: 2.0, 100000: 2.2}
SCHEDULE = {
    3000: (500000, 400000),
    10000: (500000, 400000),
    30000: (500000, 400000),
    100000: (1500000, 1200000),
}


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("n", [3000, 10000, 30000, 100000])
def test_paper_hyperparameters(setting, n):
    cfg = compose_cfg([f"model={setting}", f"data.train_size={n}"])
    w = cfg.model.loss_coeffs
    assert w.energy == 1.0
    assert w.forces == (0.0 if setting == "energy" else FORCE_WEIGHTS[n])
    assert w.charges == CHARGE_WEIGHTS.get(setting, {}).get(n, 0.0)
    assert (cfg.train.trainer.max_steps, cfg.train.lr_scheduler.beta) == SCHEDULE[n]
    assert cfg.data.split_file.endswith(f"splits/datasplits_{n}.json")
    assert cfg.core.expname == f"{setting}_{n}_seed0"


def test_unknown_train_size_falls_back_to_100k():
    cfg = compose_cfg(["model=energy_forces_charges_orbital_dxtb", "data.train_size=617554"])
    assert cfg.model.loss_coeffs.charges == 10.3
    assert cfg.train.trainer.max_steps == 1500000


@pytest.mark.parametrize(
    "exp,equivariant", [("fig1b_equivariant", True), ("fig1b_invariant", False)]
)
def test_fig1b_experiments(exp, equivariant):
    cfg = compose_cfg([f"experiment={exp}", "data.train_size=100000"])
    assert cfg.model.charges_type == "orbital_dxtb"
    assert cfg.model.model.use_equivariant_predictor is equivariant
    assert dict(cfg.model.loss_coeffs) == {"energy": 0.0, "charges": 1.0, "forces": 0.0}
    assert cfg.train.trainer.max_steps == 500000
