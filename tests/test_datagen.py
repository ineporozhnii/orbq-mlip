"""Data-generation pipeline: dxtb labels and split/pack bookkeeping."""

import json

import numpy as np
import pytest
import torch
from conftest import SAMPLE_DIR
from e3nn import o3

from orbq.data.dataset import LmdbDataset

# z-polar real d functions (xy, yz, 3z^2-r^2, xz, x^2-y^2) -> e3nn's y-polar l=2 basis
_S3 = 3**0.5 / 2
Z_POLAR_TO_E3NN = torch.tensor(
    [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, 0.0, -_S3],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, _S3, 0.0, -0.5],
    ],
    dtype=torch.double,
)


def _write_input(data, path, coords=None):
    torch.save(
        {
            "coords": (data.coords if coords is None else coords).clone(),
            "atom_types": data.atom_types.clone(),
            "charge": torch.tensor(0.0),
            "spin": torch.tensor(1.0),
        },
        path,
    )


def _blocks(labels, atom_types):
    """Split a flat label vector into (l, n_shells, block) per atom (GFN1 layout)."""
    from orbq.model.basis_set import get_basis_set, transform_basis_set
    from orbq.model.gtos import GTOs

    basis = transform_basis_set(get_basis_set("gfn1"))
    off, out = 0, []
    for z in atom_types.tolist():
        n_per_l = GTOs(**basis[z], cutoff=5.0).n_orbitals_per_L.tolist()
        for l, n in enumerate(int(v) for v in n_per_l):
            if n == 0:
                continue
            dim = n * (2 * l + 1)
            out.append((l, n, labels[off : off + dim * dim].view(dim, dim)))
            off += dim * dim
    assert off == len(labels)
    return out


@pytest.mark.dxtb
def test_dxtb_labels_match_dataset_and_transform(tmp_path):
    pytest.importorskip("dxtb")
    from orbq.datagen.dxtb_charges import calculate_dxtb

    sample = LmdbDataset(SAMPLE_DIR / "lmdb")[0]
    torch.manual_seed(0)
    rot = o3.rand_matrix().double()
    _write_input(sample, tmp_path / "orig.pt")
    _write_input(sample, tmp_path / "rot.pt", coords=(sample.coords.double() @ rot.T).float())
    for name in ("orig", "rot"):
        calculate_dxtb((str(tmp_path / f"{name}.pt"), name, str(tmp_path), 1))
    q = torch.load(tmp_path / "orig_dxtb_result.pt", weights_only=False)[
        "dxtb_charges_orbital"
    ].double()
    q_rot = torch.load(tmp_path / "rot_dxtb_result.pt", weights_only=False)[
        "dxtb_charges_orbital"
    ].double()

    # 1) recomputed labels reproduce the stored training labels
    assert torch.allclose(q.float(), sample.orbital_charges_dxtb, atol=1e-4)

    # 2) blocks transform as D Q D^T: p in e3nn's basis, d in the z-polar basis
    #    (the basis of the model's equivariant charge head)
    d_zpolar = Z_POLAR_TO_E3NN.T @ o3.Irrep(2, 1).D_from_matrix(rot) @ Z_POLAR_TO_E3NN
    wigner = {
        0: torch.ones(1, 1, dtype=torch.double),
        1: o3.Irrep(1, -1).D_from_matrix(rot),
        2: d_zpolar,
    }
    for (l, n, a), (_, _, b) in zip(
        _blocks(q, sample.atom_types), _blocks(q_rot, sample.atom_types)
    ):
        d = torch.block_diag(*[wigner[l]] * n)
        assert torch.allclose(b, d @ a @ d.T, atol=1e-4), f"l={l} block is not equivariant"


def test_prepare_split_remap_and_pack(tmp_path):
    pytest.importorskip("sklearn")
    from orbq.datagen import prepare, preprocess

    rng = np.random.default_rng(0)
    clean = tmp_path / "out" / "clean"
    clean.mkdir(parents=True)
    n_files = 16
    for i in range(n_files):
        z = torch.tensor([6, 1, 1, 8][: 2 + i % 3])
        torch.save(
            {
                "coords": torch.tensor(rng.normal(size=(len(z), 3)) * 1.5, dtype=torch.float32),
                "atom_types": z,
                "total_energy": torch.tensor(
                    float(-100 * len(z) + rng.normal()), dtype=torch.float64
                ),
                "forces": torch.zeros(len(z), 3),
                "mulliken_charges": torch.zeros(len(z)),
                "lowdin_charges": torch.zeros(len(z)),
                "dxtb_charges_orbital": torch.zeros(4),
                "dxtb_charges_atomic": torch.zeros(len(z)),
            },
            clean / f"mol_{i:04d}.pt",
        )
    # existing clean/ skips the dxtb comparison step
    prepare.main(
        [
            "--charge-dir",
            str(tmp_path),
            "--potential-dir",
            str(tmp_path),
            "--fock-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--train-sizes",
            "4",
            "8",
            "--n-val",
            "2",
            "--n-test",
            "2",
            "--num-workers",
            "2",
        ]
    )
    preprocess.main(
        [
            "--tar",
            "--data_path",
            str(tmp_path / "out" / "tar"),
            "--out_path",
            str(tmp_path / "lmdb"),
            "--num_workers",
            "1",
        ]
    )

    dataset = LmdbDataset(tmp_path / "lmdb")
    assert len(dataset) == 12  # 8 train + 2 val + 2 test packed, 4 clean files unused
    clean_files = sorted(p.name for p in clean.glob("*.pt"))
    idx = np.arange(n_files)
    np.random.seed(42)
    np.random.shuffle(idx)
    expected = {"train": idx[:8], "validation": idx[-4:-2], "test": idx[-2:]}
    splits = json.load(open(tmp_path / "out" / "splits" / "datasplits_8.json"))
    for key, clean_idx in expected.items():
        names = [dataset[j].metadata + ".pt" for j in splits[key]]
        assert names == [clean_files[i] for i in clean_idx]
