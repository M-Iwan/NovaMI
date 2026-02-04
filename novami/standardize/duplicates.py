from typing import Iterable
from copy import deepcopy

import numpy as np
import polars as pl

def mad_duplicates(df: pl.DataFrame, smiles_col: str = 'SMILES', value_col: str = 'pIC50', threshold: float = 1.0):
    """
    Automatically process duplicated entries using Median Absolute Deviation (MAD) outlier detection.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    smiles_col: str
        Column name containing the SMILES strings
    value_col: str
        Column name containing the numerical values to process.
    threshold: float
        Maximum difference between values to consider them consistent.

    Returns
    -------
    df: pl.DataFrame
    """

    def within_threshold(values: Iterable[float], threshold):
        return max(values) - min(values) <= threshold

    def mad_filter(values: Iterable[float], z_thresh: float = 3.5):
        values = np.asarray(values)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return np.ones_like(values, dtype=bool)

        modified_z = 0.6745 * (values - median) / mad
        return np.abs(modified_z) <= z_thresh

    def process_duplicates(sdf: pl.DataFrame):
        nonlocal value_col, threshold

        max_iter = len(sdf)
        iteration = 0

        while iteration < max_iter:
            values = sdf[value_col]
            if within_threshold(values, threshold):
                sdf = sdf.with_columns(pl.lit(values.mean()).alias(value_col))
                return sdf.unique()

            fsdf = sdf.filter(mad_filter(values))

            # at least one compound removed
            if len(fsdf) < len(sdf):
                sdf = fsdf
                iteration += 1
            else:
                break

        return sdf.with_columns(pl.lit(None).alias(value_col))

    int_df = deepcopy(df)
    df = int_df.select([smiles_col, value_col])

    mask = df[smiles_col].is_duplicated()

    df_unique = df.filter(~mask)
    df_duplicated = df.filter(mask)

    dfs = []
    for smiles in df_duplicated[smiles_col].unique():
        sub_df = df_duplicated.filter(pl.col(smiles_col) == smiles)
        sub_df = process_duplicates(sub_df)
        dfs.append(sub_df)

    if dfs:
        df_duplicated = pl.concat(dfs, how='vertical_relaxed')
        df = pl.concat([df_unique, df_duplicated], how='vertical_relaxed')
    else:
        df = df_unique

    df = int_df.drop(value_col).join(df.unique(), on=smiles_col, how='left')

    return df