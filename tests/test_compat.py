"""Artifacts created under the pre-release package name `mlguess` stay loadable."""

import io
import pickle

from omegaconf import OmegaConf

from orbq.common.compat import LegacyNameUnpickler, upgrade_legacy_targets
from orbq.data.data import AtomicData


def test_unpickler_maps_legacy_module():
    unpickler = LegacyNameUnpickler(io.BytesIO(pickle.dumps(None)))
    assert unpickler.find_class("mlguess.data.data", "AtomicData") is AtomicData
    assert unpickler.find_class("orbq.data.data", "AtomicData") is AtomicData


def test_upgrade_legacy_targets():
    hparams = {
        "model": OmegaConf.create({"_target_": "mlguess.model.scn.eSCN", "lmax_list": [2]}),
        "train": {"lr_scheduler": {"_target_": "mlguess.model.utils.PowerDecayScheduler"}},
        "charges_type": "orbital_dxtb",
    }
    out = upgrade_legacy_targets(hparams)
    assert out["model"]._target_ == "orbq.model.scn.eSCN"
    assert list(out["model"].lmax_list) == [2]
    assert out["train"]["lr_scheduler"]["_target_"] == "orbq.model.utils.PowerDecayScheduler"
    assert out["charges_type"] == "orbital_dxtb"
