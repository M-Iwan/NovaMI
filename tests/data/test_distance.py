import sys
import types

import numpy as np
import polars as pl
import pytest

from novami.data.distance import (
    dict_similarity,
    distance_matrix,
    group_k_neighbors_distance,
    k_largest_columns,
    k_largest_rows,
    k_neighbors_distance,
    k_smallest_columns,
    k_smallest_rows,
    list_similarity,
)


def test_distance_matrix_shape_and_validation():
    a = np.array([[0, 1], [1, 0]], dtype=float)
    b = np.array([[0, 1]], dtype=float)
    out = distance_matrix(a, b, metric="euclidean", n_jobs=1)
    assert out.shape == (2, 1)

    with pytest.raises(ValueError):
        distance_matrix(a, b, metric="invalid", n_jobs=1)


def test_k_helpers_shapes():
    arr = np.array([[0.2, 0.5, 0.1], [0.4, 0.3, 0.9]])
    assert k_smallest_rows(arr, 2).shape == (2, 2)
    assert k_smallest_columns(arr, 1).shape == (1, 3)
    assert k_largest_rows(arr, 2).shape == (2, 2)
    assert k_largest_columns(arr, 1).shape == (1, 3)


def test_k_neighbors_distance_self_comparison():
    arr = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=float)
    out = k_neighbors_distance(arr, metric="hamming", n_jobs=1, nearest_k=[1, 2], furthest_k=[1])
    assert isinstance(out, pl.DataFrame)
    assert {"Min", "Mean", "Max", "2 Nearest"}.issubset(set(out.columns))
    assert out.height == 3


def test_group_k_neighbors_distance_single_group_column():
    df = pl.DataFrame(
        {
            "X": [np.array([0, 1]), np.array([1, 0]), np.array([1, 1]), np.array([0, 0])],
            "Group": [0, 0, 1, 1],
        }
    )
    out = group_k_neighbors_distance(df, features_col="X", group_col="Group", metric="hamming", n_jobs=1)
    assert isinstance(out, pl.DataFrame)
    assert {"Scope", "Group", "Aggregation", "Values"}.issubset(set(out.columns))


def test_similarity_helpers_with_mocked_optional_deps(monkeypatch):
    fake_lev = types.SimpleNamespace(distance=lambda a, b: abs(len(a) - len(b)))
    fake_fuzz = types.SimpleNamespace(ratio=lambda a, b: 100 if a == b else 50)
    fake_rapidfuzz = types.SimpleNamespace(fuzz=fake_fuzz)
    monkeypatch.setitem(sys.modules, "Levenshtein", fake_lev)
    monkeypatch.setitem(sys.modules, "rapidfuzz", fake_rapidfuzz)

    l_matches = list_similarity("abc", ["abc", "zzz"], method="fuzzy", threshold=0.5, num_matches=2)
    d_matches = dict_similarity("abc", {"abc": "x", "zzz": "y"}, method="levenshtein", threshold=0.1, num_matches=2)
    assert l_matches[0][0] == "abc"
    assert d_matches[0][0] == "abc"

    with pytest.raises(ValueError):
        list_similarity("abc", ["abc"], method="unknown")
