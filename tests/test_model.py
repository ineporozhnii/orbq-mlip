"""Forward/backward passes and rotation invariance on the bundled sample."""

import copy

import pytest
import torch
from conftest import SETTINGS, build_model
from e3nn import o3


@pytest.mark.parametrize("setting", SETTINGS)
def test_forward_backward(setting, sample_datamodule, sample_batch):
    model = build_model(setting, sample_datamodule.metadata)
    n_params = sum(p.numel() for p in model.parameters())
    assert 6.5e6 < n_params < 6.9e6  # ~6.7M parameters (paper)
    loss, pred_e, target_e, pred_q, target_q, pred_f, target_f = model(sample_batch)
    assert torch.isfinite(loss)
    assert pred_e.shape == target_e.shape
    assert (pred_f is not None) == ("forces" in setting)
    if "charges" in setting:
        assert pred_q.shape == target_q.shape
        assert model.pred_charges_diag.shape == model.target_charges_diag.shape
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rotation_invariance(sample_datamodule, sample_batch):
    """Energies, atomic charges and orbital-block traces are rotation invariant."""
    model = build_model("energy_forces_charges_orbital_dxtb", sample_datamodule.metadata).eval()
    torch.manual_seed(1)
    rot = o3.rand_matrix()
    rotated = copy.deepcopy(sample_batch)
    rotated.coords = sample_batch.coords @ rot.T
    with torch.no_grad():
        e1, _, q1, a1, _ = model.model(sample_batch)
        e2, _, q2, a2, _ = model.model(rotated)

    def rel(x, y):
        return ((x - y).abs().max() / y.abs().max()).item()

    assert rel(e2, e1) < 1e-3
    assert rel(a2, a1) < 1e-3
    offsets = [0] + torch.cumsum(model.max_matrix_dim_per_L, 0).tolist()
    for l in range(model.Lmax + 1):
        if int(model.max_n_orbitals_per_L[l]) != 1:
            continue
        w = 2 * l + 1
        tr1 = q1[:, offsets[l] : offsets[l + 1]].reshape(-1, w, w).diagonal(dim1=1, dim2=2).sum(-1)
        tr2 = q2[:, offsets[l] : offsets[l + 1]].reshape(-1, w, w).diagonal(dim1=1, dim2=2).sum(-1)
        assert rel(tr2, tr1) < 1e-3
