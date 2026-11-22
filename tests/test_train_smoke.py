"""End-to-end CPU training run through the Hydra entry point."""

import subprocess
import sys


def test_train_smoke(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "orbq.scripts.train",
        "data=sample",
        "train=smoke",
        "model=energy_forces_charges_orbital_dxtb",
        f"core.storage_dir={tmp_path / 'run'}",
    ]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert (tmp_path / "run" / "last.ckpt").exists()
    assert "mae/test_energy" in proc.stdout + proc.stderr
