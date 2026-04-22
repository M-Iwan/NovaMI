import json
import sys
import types

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest
from rdkit import Chem

import novami.data.descriptors as desc


class _FakeHidden:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def mean(self, dim=1):
        return _FakeHidden(np.mean(self._arr, axis=dim))

    def squeeze(self):
        return self

    def numpy(self):
        return self._arr


class _FakeModelOutput:
    def __init__(self, arr):
        self.last_hidden_state = _FakeHidden(arr)


class _FakeModel:
    def eval(self):
        return None

    def __call__(self, **tokens):
        return _FakeModelOutput([[[1.11111, 2.22222, 3.33333]]])


class _FakeTokenizer:
    def __call__(self, smiles, return_tensors, padding, truncation, max_length):
        return {"input_ids": [1, 2, 3]}


def _mock_transformers(monkeypatch):
    fake = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: _FakeTokenizer()),
        AutoModel=types.SimpleNamespace(from_pretrained=lambda _: _FakeModel()),
        logging=types.SimpleNamespace(set_verbosity_error=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_smiles_2_klek_and_dataframe_2_klek(monkeypatch):
    smarts = [Chem.MolFromSmarts("C"), Chem.MolFromSmarts("O")]
    monkeypatch.setattr(desc.joblib, "load", lambda _: smarts)

    single = desc.smiles_2_klek("CCO")
    many = desc.smiles_2_klek(["CCO", "CC"])
    assert isinstance(single, np.ndarray)
    assert len(many) == 2

    df = pl.DataFrame({"SMILES": ["CCO", "CCO", "CC"]})
    out = desc.dataframe_2_klek(df, n_jobs=1, batch_size=1)
    assert "Klek" in out.columns
    assert out["Klek"].is_null().sum() == 0


def test_smiles_2_chemberta_and_dataframe_2_chemberta(monkeypatch):
    _mock_transformers(monkeypatch)
    monkeypatch.setattr(desc.joblib, "load", lambda p: _FakeTokenizer() if "tokenizer" in str(p) else _FakeModel())
    monkeypatch.setattr(desc.os.path, "isfile", lambda p: True)

    emb = desc.smiles_2_chemberta("CCO", decimals=3)
    assert isinstance(emb, np.ndarray)
    assert emb.size > 0

    df = pl.DataFrame({"SMILES": ["CCO", "CCN"]})
    out = desc.dataframe_2_chemberta(df, n_jobs=1, batch_size=1)
    assert "ChemBERTa" in out.columns
    assert out.height == 2


def test_get_chemberta_download_and_dump(monkeypatch):
    _mock_transformers(monkeypatch)
    dumped = []
    monkeypatch.setattr(desc.joblib, "dump", lambda obj, path: dumped.append(str(path)))
    desc.get_chemberta()
    assert any("ChemBERTa-tokenizer.joblib" in path for path in dumped)
    assert any("ChemBERTa-model.joblib" in path for path in dumped)


def test_smiles_2_mapc_and_dataframe_2_mapc(monkeypatch):
    fake_mapchiral_mod = types.SimpleNamespace(encode=lambda mol, max_radius, n_permutations: np.array([1, 0, 1]))
    monkeypatch.setitem(sys.modules, "mapchiral.mapchiral", fake_mapchiral_mod)
    monkeypatch.setitem(sys.modules, "mapchiral", types.SimpleNamespace(mapchiral=fake_mapchiral_mod))

    out_single = desc.smiles_2_mapc("CCO", radius=2, nbits=8)
    assert isinstance(out_single, np.ndarray)

    df = pl.DataFrame({"SMILES": ["CCO", "CCN"]})
    out_df = desc.dataframe_2_mapc(df, n_jobs=1, batch_size=1)
    assert "MAPC" in out_df.columns


def test_dataframe_2_mordred_with_mocked_io(tmp_path, monkeypatch):
    paths = {
        "python": "python",
        "wrapper": "wrapper.py",
        "input": str(tmp_path / "in.tsv"),
        "output": str(tmp_path / "out.joblib"),
    }
    source = tmp_path / "mordred_paths.json"
    source.write_text(json.dumps(paths))

    monkeypatch.setattr(desc, "write_pd", lambda dataframe, path: None)
    monkeypatch.setattr(desc, "os", types.SimpleNamespace(system=lambda command: 0, path=desc.os.path))
    mock_out = pd.DataFrame({"SMILES": ["CCO"], "Mordred": [np.array([1.11111, 2.22222])]})
    monkeypatch.setattr(desc, "read_pd", lambda path: mock_out.copy())

    df = pl.DataFrame({"SMILES": ["CCO"]})
    out = desc.dataframe_2_mordred(df, path_source=str(source), decimals=3)
    assert "Mordred" in out.columns
    assert np.asarray(out["Mordred"][0]).size == 2


def test_dataframe_2_cddd_with_mocked_io(tmp_path, monkeypatch):
    paths = {
        "python": "python",
        "wrapper": "wrapper.py",
        "input": str(tmp_path / "in.tsv"),
        "output": str(tmp_path / "out.joblib"),
        "model": str(tmp_path / "model"),
    }
    source = tmp_path / "cddd_paths.json"
    source.write_text(json.dumps(paths))

    monkeypatch.setattr(desc, "os", types.SimpleNamespace(system=lambda command: 0, path=desc.os.path))
    mock_out = pd.DataFrame({"SMILES": ["CCO"], "CDDD": [np.array([0.1, 0.2, 0.3])]})
    monkeypatch.setattr(joblib, "load", lambda path: mock_out.copy())

    df = pl.DataFrame({"SMILES": ["CCO"]})
    out = desc.dataframe_2_cddd(df, cddd_paths=str(source), decimals=3)
    assert "CDDD" in out.columns
    assert np.asarray(out["CDDD"][0]).size == 3
