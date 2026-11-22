"""orbq: multitask MLIPs with semiempirical orbital charges."""

from pathlib import Path

from omegaconf import OmegaConf

__version__ = "0.1.0"

# ${orbq_root:} resolves to the source tree root (used for bundled sample data)
OmegaConf.register_new_resolver(
    "orbq_root", lambda: str(Path(__file__).resolve().parents[1]), replace=True
)
