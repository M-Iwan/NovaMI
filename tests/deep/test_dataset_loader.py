import warnings

import numpy as np
import polars as pl
import pytest
import torch

import novami.deep.loader as loader_mod
from novami.deep.dataset import MMBatch, MMDataset
from novami.deep.loader import MMLoader
from novami.deep.models import MMTUnit


def _make_base_df(include_weights: bool = True, include_graph: bool = True, include_group: bool = False):
    data = {
        "tokens": pl.Series(
            "tokens",
            [
                (torch.tensor([1, 2, 0, 0]), 2),
                (torch.tensor([3, 4, 5, 0]), 3),
                (torch.tensor([6, 0, 0, 0]), 1),
            ],
            dtype=pl.Object,
        ),
        "desc": pl.Series(
            "desc",
            [
                torch.tensor([0.1, 0.2], dtype=torch.float32),
                torch.tensor([0.3, 0.4], dtype=torch.float32),
                torch.tensor([0.5, 0.6], dtype=torch.float32),
            ],
            dtype=pl.Object,
        ),
        "target": pl.Series(
            "target",
            [
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
            ],
            dtype=pl.Object,
        ),
    }
    if include_group:
        data["project"] = pl.Series(
            "project",
            [
                np.array(["Small"], dtype=object),
                np.array(["Small", "Acid"], dtype=object),
                np.array(["Acid"], dtype=object),
            ],
            dtype=pl.Object,
        )
    if include_weights:
        data["weight"] = pl.Series(
            "weight",
            [
                torch.tensor([0.5], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
            ],
            dtype=pl.Object,
        )
    if include_graph:
        data["graph_data"] = ["g1", "g2", "g3"]

    return pl.DataFrame(data)


def test_mmdataset_validates_config_missing_and_unsupported():
    df = _make_base_df()
    with pytest.raises(ValueError):
        MMDataset(df, config={"missing": "descriptor", "target": "target"})
    with pytest.raises(ValueError):
        MMDataset(df, config={"desc": "unknown", "target": "target"})


def test_mmdataset_requires_exactly_one_target_and_one_feature():
    df = _make_base_df()
    with pytest.raises(ValueError):
        MMDataset(df, config={"desc": "descriptor"})
    with pytest.raises(ValueError):
        MMDataset(df, config={"target": "target", "weight": "sample_weight"})


def test_mmdataset_renames_target_and_weight_columns():
    df = _make_base_df(include_weights=True)
    config = {
        "tokens": "string",
        "desc": "descriptor",
        "target": "target",
        "weight": "sample_weight",
    }
    ds = MMDataset(df, config=config)

    assert len(ds) == 3
    assert ds.get_target_column() == "target"
    assert ds.has_sample_weights() is True
    assert ds.has_groups() is False
    assert "y_true" in ds.df.columns
    assert "y_wgts" in ds.df.columns
    assert ds.get_feature_columns() == ["tokens", "desc"]

    sample = ds[0]
    assert {"tokens", "desc", "y_true", "y_wgts"}.issubset(set(sample.keys()))


def test_mmbatch_helpers_and_to_cpu():
    batch = MMBatch(
        data={
            "desc": torch.tensor([[1.0, 2.0]]),
            "tokens": (torch.tensor([[1, 2, 0]]), torch.tensor([2]), torch.tensor([[True, True, False]])),
            "group": [1],
        },
        config={"desc": "descriptor", "tokens": "string", "group": "group"},
    )
    assert "desc" in batch
    assert batch.get_modality("desc") == "descriptor"
    assert set(batch.get_by_modality("string").keys()) == {"tokens"}
    moved = batch.to("cpu")
    assert isinstance(moved["desc"], torch.Tensor)
    assert moved["desc"].device.type == "cpu"


def test_mmloader_collates_string_descriptor_target_and_default_weights():
    df = _make_base_df(include_weights=False, include_graph=False)
    config = {"tokens": "string", "desc": "descriptor", "target": "target"}
    ds = MMDataset(df, config=config)
    loader = MMLoader(ds, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert isinstance(batch, MMBatch)
    assert "tokens" in batch and "desc" in batch and "y_true" in batch and "y_wgts" in batch

    stacked_tokens, lengths, attention_mask = batch["tokens"]
    assert stacked_tokens.shape == (2, 4)
    assert lengths.shape == (2,)
    assert attention_mask.shape == (2, 4)
    assert batch["desc"].shape == (2, 2)
    assert torch.all(batch["y_wgts"] == 1.0)


def test_mmdataset_renames_group_column_and_getitem():
    df = _make_base_df(include_weights=False, include_graph=False, include_group=True)
    ds = MMDataset(
        df,
        config={"tokens": "string", "target": "target", "project": "group"},
    )
    assert ds.get_group_column() == "project"
    assert "group" in ds.df.columns
    assert "project" not in ds.df.columns
    assert ds.has_groups() is True
    row = ds[0]
    assert "group" in row
    assert np.asarray(row["group"]).tolist() == ["Small"]


def test_mmloader_uses_sample_weights_when_present():
    df = _make_base_df(include_weights=True, include_graph=False)
    config = {"tokens": "string", "desc": "descriptor", "target": "target", "weight": "sample_weight"}
    ds = MMDataset(df, config=config)
    loader = MMLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert torch.allclose(batch["y_wgts"].reshape(-1), torch.tensor([0.5, 1.0], dtype=torch.float32))


def test_mmloader_graph_collation_success(monkeypatch):
    df = _make_base_df(include_weights=False, include_graph=True)
    config = {"graph_data": "graph", "target": "target"}
    ds = MMDataset(df, config=config)
    loader = MMLoader(ds, batch_size=2, shuffle=False)

    class _FakeGB:
        @staticmethod
        def from_data_list(graphs):
            return {"batched_graphs": graphs}

    monkeypatch.setattr(loader_mod, "GeometricBatch", _FakeGB)
    batch = next(iter(loader))
    assert batch["graph_data"]["batched_graphs"] == ["g1", "g2"]


def test_mmloader_graph_collation_fallback_warns(monkeypatch):
    df = _make_base_df(include_weights=False, include_graph=True)
    config = {"graph_data": "graph", "target": "target"}
    ds = MMDataset(df, config=config)
    loader = MMLoader(ds, batch_size=2, shuffle=False)

    class _BrokenGB:
        @staticmethod
        def from_data_list(graphs):
            raise RuntimeError("boom")

    monkeypatch.setattr(loader_mod, "GeometricBatch", _BrokenGB)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        batch = next(iter(loader))
    assert isinstance(batch["graph_data"], list)
    assert any("Failed to batch graphs" in str(w.message) for w in records)


def test_mmloader_batches_group_list():
    df = _make_base_df(include_weights=False, include_graph=False, include_group=True)
    ds = MMDataset(
        df,
        config={"tokens": "string", "target": "target", "project": "group"},
    )
    loader = MMLoader(ds, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert "group" in batch
    assert len(batch["group"]) == 3
    assert isinstance(batch["group"][1], np.ndarray)


def test_mmtunit_mw_loss_partial_per_group():
    class _MinimalMMT(MMTUnit):
        def forward(self, batch):
            return torch.zeros(batch["y_true"].shape[0], self.num_task, device=self.device)

        @staticmethod
        def score(y_true, y_pred, y_wgts):
            return {}

    m = _MinimalMMT(device="cpu", num_task=1)
    m.set_loss_function(torch.nn.MSELoss, {"reduction": "none"})
    y_pred = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y_true = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    y_wgts = torch.ones_like(y_true)
    group = [np.array(["A"], dtype=object), np.array(["A", "B"], dtype=object)]
    out = m.mw_loss(y_pred, y_true, y_wgts, group=group)
    assert "Group" in out
    assert set(out["Group"].keys()) == {"A", "B"}
    norm = MMTUnit.normalize_mw_loss(out)
    np.testing.assert_allclose(float(norm["Group"]["A"]), 2.0, rtol=1e-5)
    np.testing.assert_allclose(float(norm["Group"]["B"]), 4.0, rtol=1e-5)
