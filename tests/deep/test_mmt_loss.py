"""
Runtime validation of :class:`novami.deep.models.MMTUnit` masked weighted loss:
NaN targets (sparse), multitask tensors, and per-group aggregation.
"""

from functools import reduce
from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch import nn

from novami.deep.models import MMTUnit


class _LossProbe(MMTUnit):
    """Minimal concrete unit for exercising ``mw_loss`` without a full graph model."""

    def forward(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def score(y_true: np.ndarray, y_pred: np.ndarray, y_wgts: np.ndarray) -> Dict[str, float]:
        return {}


def _reference_mw_loss(
    loss_fn: nn.Module, y_pred: torch.Tensor, y_true: torch.Tensor, y_wgts: torch.Tensor
) -> torch.Tensor:
    mask = ~torch.isnan(y_true)
    y_filled = torch.nan_to_num(y_true, nan=0.0)
    raw = loss_fn(y_pred, y_filled)
    return raw * (y_wgts * mask)


@pytest.fixture
def probe() -> _LossProbe:
    m = _LossProbe(device="cpu", num_task=1)
    m.set_loss_function(nn.MSELoss, {"reduction": "none"})
    return m


def test_mw_loss_total_matches_masked_weighted_sum(probe: _LossProbe):
    """H1: ``Total`` tuple matches sum of per-element masked weighted loss."""
    y_pred = torch.tensor([[0.0, 2.0], [1.0, 1.0]], dtype=torch.float32)
    y_true = torch.tensor([[0.0, float("nan")], [float("nan"), 3.0]], dtype=torch.float32)
    y_wgts = torch.tensor([[1.0, 2.0], [0.5, 1.0]], dtype=torch.float32)
    ref = _reference_mw_loss(probe.loss_fn, y_pred, y_true, y_wgts)
    out = probe.loss(y_pred, y_true, y_wgts)
    tl, tw = out["Total"]
    assert torch.allclose(tl, ref.sum())
    assert torch.allclose(tw, (y_wgts * (~torch.isnan(y_true))).sum())


def test_mw_loss_per_task_matches_column_sums(probe: _LossProbe):
    """H2: ``Task`` first tensor equals column sums of masked weighted loss."""
    y_pred = torch.randn(5, 3)
    y_true = y_pred.clone()
    y_true[0, 1] = float("nan")
    y_true[2, 0] = float("nan")
    y_wgts = torch.ones_like(y_pred)
    ref = _reference_mw_loss(probe.loss_fn, y_pred, y_true, y_wgts)
    out = probe.loss(y_pred, y_true, y_wgts)
    pt, ptw = out["Task"]
    assert torch.allclose(pt, ref.sum(dim=0))
    assert torch.allclose(ptw, (y_wgts * (~torch.isnan(y_true))).sum(dim=0))


def test_mw_loss_group_partial_sums(probe: _LossProbe):
    """H3: Each group label's (loss_sum, weight_sum) matches row-masked totals."""
    y_pred = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    y_true = torch.tensor([[0.0, 1.0], [1.0, float("nan")], [2.0, 2.0]], dtype=torch.float32)
    y_wgts = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    group = [np.array(["A", "B"]), ["A"], np.array(["B"])]
    ref = _reference_mw_loss(probe.loss_fn, y_pred, y_true, y_wgts)
    out = probe.loss(y_pred, y_true, y_wgts, group=group)
    assert "Group" in out
    for label in ("A", "B"):
        gl, gw = out["Group"][label]
        rows = []
        for i, tags in enumerate(group):
            flat = np.asarray(tags, dtype=object).ravel()
            tagset = {
                str(x).strip()
                for x in flat
                if x is not None and str(x).strip() and str(x).strip().lower() != "nan"
            }
            if label in tagset:
                rows.append(i)
        rm = torch.zeros(y_pred.shape[0], 1, 1, dtype=y_pred.dtype)
        for i in rows:
            rm[i] = 1.0
        exp_l = (ref * rm).sum()
        exp_w = (y_wgts * (~torch.isnan(y_true)) * rm.squeeze(-1)).sum()
        assert torch.allclose(gl, exp_l)
        assert torch.allclose(gw, exp_w)


def test_combine_mw_loss_merges_group_keys(probe: _LossProbe):
    """H4: ``combine_mw_loss`` unions group labels and sums (loss, weight) pairs."""
    y = torch.zeros(1, 1)
    w = torch.ones_like(y)
    a = probe.loss(y, y, w, group=[["x"]])
    b = probe.loss(y + 1.0, y, w, group=[["y"]])
    c = MMTUnit.combine_loss(a, b)
    assert set(c["Group"].keys()) == {"x", "y"}
    d = MMTUnit.combine_loss(b, a)
    assert set(d["Group"].keys()) == {"x", "y"}


def test_normalize_mw_loss_division(probe: _LossProbe):
    """H5: Normalized totals match tensor division (and zero-weight groups map to 0)."""
    y_pred = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    y_true = torch.tensor([[0.0, float("nan")]], dtype=torch.float32)
    y_wgts = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    batch = probe.loss(y_pred, y_true, y_wgts, group=[["z"]])
    z_l, z_w = batch["Group"]["z"]
    # Force a zero-weight group entry by hand (simulates merged epoch edge case)
    batch["Group"]["empty"] = (torch.tensor(0.0), torch.tensor(0.0))
    norm = MMTUnit.normalize_loss(batch)
    tl, tw = batch["Total"]
    assert np.isclose(norm["Total"], float((tl / tw).cpu().numpy()))
    task_num = (batch["Task"][0] / batch["Task"][1]).detach().cpu().numpy()
    assert np.allclose(norm["Task"], np.round(task_num, 5), equal_nan=True)
    assert norm["Group"]["empty"] == 0.0
    assert np.isclose(norm["Group"]["z"], float((z_l / z_w).cpu().numpy()))


def test_group_skipped_when_batch_length_mismatch(probe: _LossProbe):
    y_pred = torch.zeros(3, 1)
    y_true = torch.zeros_like(y_pred)
    y_wgts = torch.ones_like(y_pred)
    out = probe.loss(y_pred, y_true, y_wgts, group=[["a"], ["b"]])
    assert "Group" not in out


def test_reduce_combine_matches_single_concat(probe: _LossProbe):
    batches = []
    for t in range(3):
        yp = torch.full((2, 2), float(t), dtype=torch.float32)
        yt = yp.clone()
        yt[0, 0] = float("nan") if t == 1 else yt[0, 0]
        yw = torch.ones_like(yp)
        batches.append(probe.loss(yp, yt, yw))
    merged = reduce(MMTUnit.combine_loss, batches)
    cat_pred = torch.cat([torch.full((2, 2), float(t), dtype=torch.float32) for t in range(3)], dim=0)
    cat_true = cat_pred.clone()
    cat_true[2, 0] = float("nan")
    cat_w = torch.ones_like(cat_pred)
    direct = probe.loss(cat_pred, cat_true, cat_w)
    assert torch.allclose(merged["Total"][0], direct["Total"][0])
    assert torch.allclose(merged["Total"][1], direct["Total"][1])
    assert torch.allclose(merged["Task"][0], direct["Task"][0])
    assert torch.allclose(merged["Task"][1], direct["Task"][1])


def test_bce_logits_binary_multitask(probe: _LossProbe):
    probe.set_loss_function(nn.BCEWithLogitsLoss, {"reduction": "none"})
    y_pred = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], dtype=torch.float32)
    y_true = torch.tensor([[0.0, 1.0], [1.0, float("nan")]], dtype=torch.float32)
    y_wgts = torch.tensor([[1.0, 0.5], [1.0, 1.0]], dtype=torch.float32)
    ref = _reference_mw_loss(probe.loss_fn, y_pred, y_true, y_wgts)
    out = probe.loss(y_pred, y_true, y_wgts)
    assert torch.allclose(out["Total"][0], ref.sum())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
def test_group_masks_on_device(probe: _LossProbe, device: str):
    probe.to(device)
    y_pred = torch.tensor([[0.0], [1.0]], device=device)
    y_true = y_pred.clone()
    y_wgts = torch.ones_like(y_pred)
    out = probe.loss(y_pred, y_true, y_wgts, group=[["g"], ["g"]])
    gl, gw = out["Group"]["g"]
    assert gl.device.type == device
    assert gw.device.type == device
