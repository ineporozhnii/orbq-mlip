"""Backward compatibility with artifacts created before the package rename.

The package was called ``mlguess`` before its public release as ``orbq``.
Preprocessed LMDB datasets pickle ``mlguess.data.data.AtomicData`` objects and
training checkpoints store Hydra ``_target_`` strings such as
``mlguess.model.module.MultitaskLightningModule``. The helpers below map those
names to ``orbq`` so old datasets and checkpoints keep working.
"""

import io
import pickle
import types

from omegaconf import DictConfig, ListConfig, OmegaConf

LEGACY_PACKAGE = "mlguess"
PACKAGE = "orbq"


def _rename_module(module: str) -> str:
    if module == LEGACY_PACKAGE or module.startswith(LEGACY_PACKAGE + "."):
        return PACKAGE + module[len(LEGACY_PACKAGE) :]
    return module


class LegacyNameUnpickler(pickle.Unpickler):
    """Unpickler that resolves ``mlguess.*`` classes to ``orbq.*``."""

    def find_class(self, module, name):
        return super().find_class(_rename_module(module), name)


def loads(data: bytes):
    """Drop-in replacement for ``pickle.loads`` that understands legacy names."""
    return LegacyNameUnpickler(io.BytesIO(data)).load()


# Module-like object for ``torch.load(..., pickle_module=legacy_pickle_module)``.
legacy_pickle_module = types.SimpleNamespace(
    Unpickler=LegacyNameUnpickler,
    load=lambda f, **kw: LegacyNameUnpickler(f, **kw).load(),
    loads=loads,
    __name__="orbq_legacy_pickle",
)


def upgrade_legacy_targets(obj):
    """Recursively rewrite ``mlguess.*`` strings (e.g. Hydra ``_target_``) to ``orbq.*``."""
    if isinstance(obj, (DictConfig, ListConfig)):
        container = OmegaConf.to_container(obj, resolve=False)
        return OmegaConf.create(upgrade_legacy_targets(container))
    if isinstance(obj, dict):
        return type(obj)((k, upgrade_legacy_targets(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [upgrade_legacy_targets(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(LEGACY_PACKAGE + "."):
        return _rename_module(obj)
    return obj
