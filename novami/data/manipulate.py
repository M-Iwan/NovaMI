from typing import Union, List, Tuple, Optional, Iterable
import pickle
import os

import numpy as np
import pandas as pd
import polars as pl
from rdkit import DataStructs


def bin_data(data: Iterable[float], n_bins: int = 5):
    """
    Assign entries in data into bins.

    Parameters
    ----------
    data: Iterable[float]
        E.g. list, 1D np.array, pd/pl.Series
    n_bins: int
        Into how many bins data should be assigned

    Returns
    -------
    bins: List[int]
    """
    quantiles = list(np.quantile(data, q=np.linspace(0, 1, n_bins+1)[1:-1]))

    def to_bin(value: float, qnts: List[float]):
        for idx, quantile in enumerate(qnts):
            if value < quantile:
                return idx + 1
        return len(quantiles) + 1

    bins = [to_bin(value, qnts=quantiles) for value in data]
    return bins


def is_valid_fingerprint(fingerprint: np.ndarray) -> bool:
    """
    Check if numpy array contains only [0,1] values

    Parameters
    ----------
    fingerprint: np.ndarray
        The array to check

    Returns
    -------
    bool
    """

    return np.all(np.isin(fingerprint, [0, 1]))


def embeddings_to_rdkit(embeddings: Iterable[np.ndarray]) -> list:
    """
    Convert numpy arrays to RDKit's native DataStructs instance

    Parameters
    ----------
    embeddings: Iterable[np.ndarray]
        Embeddings to convert (e.g. List[np.ndarray])

    Returns
    -------
    fingerprints: List
    """

    fingerprints = []
    for emb in embeddings:
        fingerprint = ndarray_to_binary_string(emb)
        fingerprints.append(DataStructs.CreateFromBitString(fingerprint))
    return fingerprints


def ndarray_to_binary_string(array: np.ndarray) -> str:
    """
    Convert a binary fingerprint represented as numpy array string.

    Parameters
    ----------
    array: np.ndarray
        The array to convert

    Returns
    -------
    str
    """

    if not len(array) or not is_valid_fingerprint(array):
        raise ValueError(
            "Invalid fingerprint array. Expected binary Morgan fingerprint with 0s and 1s."
        )
    return "".join(array.astype(str).tolist())
