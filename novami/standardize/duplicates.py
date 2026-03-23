from typing import Iterable, List
from copy import deepcopy

import numpy as np
import polars as pl


def check_unit_error(values: List[float]) -> bool:
    """
    Check if for a given SMILES, there exists an entry differing by 3/6/9 log10 units.
    Return True if Unit Error was detected, False otherwise.
    """
    if not isinstance(values, list):
        return False
    if len(values) == 1:
        return False
    for i in range(len(values) - 1):
        v = values[i]
        ov = set(values[i:])
        vs = {v - 9, v - 6, v - 3, v + 3, v + 6, v + 9}
        if len(vs.intersection(ov)) > 0:
            return True
    return False


def mad_duplicates(df: pl.DataFrame, smiles_col: str = 'SMILES', value_col: str = 'pIC50',
                   range_threshold: float = 1.0, z_threshold: float = 3.5) -> pl.DataFrame:
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
    range_threshold: float
        Maximum difference between values to consider them consistent. Default is 1.0
    z_threshold: float
        Maximum threshold for MAD outlier detection. Default is 3.5

    Returns
    -------
    df: pl.DataFrame
    """

    def within_threshold(values: Iterable[float], r_thresh):
        return max(values) - min(values) <= r_thresh

    def mad_filter(values: Iterable[float], z_thresh: float = 3.5):
        values = np.asarray(values)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return np.ones_like(values, dtype=bool)

        modified_z = 0.6745 * (values - median) / mad
        return np.abs(modified_z) <= z_thresh

    def process_duplicates(sdf: pl.DataFrame):
        nonlocal value_col, range_threshold, z_threshold

        max_iter = len(sdf)
        iteration = 0

        while iteration < max_iter:
            values = sdf[value_col]
            if within_threshold(values, range_threshold):
                sdf = sdf.with_columns(pl.lit(values.mean()).alias(value_col))
                return sdf.unique()

            fsdf = sdf.filter(mad_filter(values, z_threshold))

            # at least one compound removed
            if len(fsdf) < len(sdf):
                sdf = fsdf
                iteration += 1
            else:
                break

        return sdf.with_columns(pl.lit(None).alias(value_col))

    int_df = deepcopy(df).select([smiles_col, value_col])

    mask = int_df[smiles_col].is_duplicated()

    df_unique = int_df.filter(~mask)
    df_duplicated = int_df.filter(mask)

    dfs = []
    for smiles in df_duplicated[smiles_col].unique():
        sub_df = df_duplicated.filter(pl.col(smiles_col) == smiles)
        sub_df = process_duplicates(sub_df)
        dfs.append(sub_df)

    if dfs:
        df_duplicated = pl.concat(dfs, how='vertical_relaxed')
        out_df = pl.concat([df_unique, df_duplicated], how='vertical_relaxed')
    else:
        out_df = df_unique

    df = df.drop(value_col).join(out_df.unique(), on=smiles_col, how='left')

    return df