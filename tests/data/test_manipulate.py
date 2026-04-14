import numpy as np
import pytest

from novami.data.manipulate import (
    bin_data,
    embeddings_to_rdkit,
    is_valid_fingerprint,
    ndarray_to_binary_string,
)


def test_bin_data_returns_expected_bin_count():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    bins = bin_data(data, n_bins=3)
    assert len(bins) == len(data)
    assert all(1 <= b <= 3 for b in bins)


def test_is_valid_fingerprint():
    assert is_valid_fingerprint(np.array([0, 1, 1, 0]))
    assert not is_valid_fingerprint(np.array([0, 2, 1]))


def test_ndarray_to_binary_string():
    assert ndarray_to_binary_string(np.array([1, 0, 1, 0])) == "1010"


def test_ndarray_to_binary_string_raises_for_invalid():
    with pytest.raises(ValueError):
        ndarray_to_binary_string(np.array([1, 2, 0]))


def test_embeddings_to_rdkit_returns_rdkit_bitvectors():
    fps = embeddings_to_rdkit([np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])])
    assert len(fps) == 2
    assert all(hasattr(fp, "GetNumBits") for fp in fps)
