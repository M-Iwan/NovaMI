from typing import Iterable, List
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
        ov = set(values[i+1:])
        vs = {v - 9, v - 6, v - 3, v + 3, v + 6, v + 9}
        if len(vs.intersection(ov)) > 0:
            return True
    return False


def mad_duplicates(df: pl.DataFrame, dd_col: str = 'dd_key', smiles_col: str = 'SMILES', value_col: str = 'pIC50',
                   range_threshold: float = 1.0, z_threshold: float = 3.5):
    """
    Automatically process duplicated entries using Median Absolute Deviation (MAD) outlier detection.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    dd_col: str
        Column name used to group related measurements together
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
        The input DataFrame with `value_col` replaced by the deduplicated/
        aggregated value
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
            values = sdf[value_col].to_numpy()
            if within_threshold(values, range_threshold):
                sdf = sdf.with_columns(pl.lit(values.mean()).alias(value_col))
                return sdf.unique()

            mask = mad_filter(values, z_threshold)
            fsdf = sdf.filter(pl.Series(mask))

            if len(fsdf) < len(sdf):
                sdf = fsdf
                iteration += 1
            else:
                break

        return sdf.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(value_col)
        )

    int_df = df.select([dd_col, smiles_col, value_col])

    mask = int_df[dd_col].is_duplicated()

    df_unique = int_df.filter(~mask)
    df_duplicated = int_df.filter(mask)

    dfs = []
    for dd_key in df_duplicated[dd_col].unique():
        sub_df = df_duplicated.filter(pl.col(dd_col) == dd_key)
        sub_df = process_duplicates(sub_df)
        dfs.append(sub_df)

    if dfs:
        df_duplicated = pl.concat(dfs, how='vertical_relaxed')
        out_df = pl.concat([df_unique, df_duplicated], how='vertical_relaxed')
    else:
        out_df = df_unique

    df = df.drop(value_col).join(
        out_df.select([dd_col, value_col]).unique(subset=dd_col),
        on=dd_col,
        how='left',
    )

    return df


def mad_censored_duplicates(df: pl.DataFrame, dd_col: str = 'dd_key', smiles_col: str = 'SMILES', value_col: str = 'p_value',
                            relation_col: str = 'standard_relation', range_threshold: float = 1.0, z_threshold: float = 3.5):
    """
    Automatically process duplicated/grouped entries that may contain censored
    (">", "<") as well as exact ("=") measurements, using Median Absolute
    Deviation (MAD) outlier detection to collapse each group down to a single
    (value, relation) pair.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    dd_col: str
        Column name used to group related measurements together
    smiles_col: str
        Column name containing the SMILES strings.
    value_col: str
        Column name containing the numerical values to process.
    relation_col: str
        Column name containing the relation for each value ("=", ">", "<").
    range_threshold: float
        Maximum difference between values to consider them consistent.
        Should correspond to the expected experimental error. Default is 1.0
    z_threshold: float
        Maximum threshold for MAD outlier detection. Default is 3.5

    Returns
    -------
    df: pl.DataFrame
        The input DataFrame with `value_col` and `relation_col` replaced by
        their deduplicated/aggregated counterparts
    """

    def process_subset(values: np.ndarray, relations: np.ndarray, r_thr: float, z_thr: float = 3.5):
        """
        Aggregate bioactivity measurements with exact and/or censored values.
        Permitted relations are: "=", ">", "<".
        """

        def within_threshold(_values: np.ndarray, _r_thr: float):
            return (_values.max() - _values.min()) <= _r_thr

        def mad_filter(_values: np.ndarray, _z_thr: float):
            median = np.median(_values)
            mad = np.median(np.abs(_values - median))
            if mad == 0:
                return np.ones_like(_values, dtype=bool)
            modified_z = 0.6745 * (_values - median) / mad
            return np.abs(modified_z) <= _z_thr

        def _sort(_values: np.ndarray, _relations: np.ndarray):
            _ex, _gt, _lt = [], [], []
            for idx, rel in enumerate(_relations):
                if rel == "=":
                    _ex.append(_values[idx])
                elif rel == "<":
                    _lt.append(_values[idx])
                elif rel == ">":
                    _gt.append(_values[idx])
                else:
                    raise ValueError(f"Relation not allowed: {rel}")
            return (
                np.array(_ex, dtype=np.float64),
                np.array(_gt, dtype=np.float64),
                np.array(_lt, dtype=np.float64),
            )

        def _process(_values: np.ndarray, _r_thr: float, _z_thr: float, rel: str):
            _values = np.asarray(_values, dtype=np.float64)

            def _agg(_values: np.ndarray, _rel: str):
                if _rel == "=":
                    return np.median(_values)
                elif _rel == "<":
                    return np.min(_values)
                elif _rel == ">":
                    return np.max(_values)
                else:
                    raise ValueError(f"Relation not allowed: {_rel}")

            if len(_values) == 0:
                return None
            if len(_values) == 1:
                return _agg(_values, rel)
            if within_threshold(_values, _r_thr):
                return _agg(_values, rel)
            if len(_values) == 2:
                return None

            while len(_values) >= 3:
                prev_size = len(_values)
                _values = _values[mad_filter(_values, _z_thr)]

                if len(_values) == 0:
                    return None
                if len(_values) == 1:
                    return _values[0]
                if len(_values) == 2:
                    return _agg(_values, rel) if within_threshold(_values, _r_thr) else None
                if within_threshold(_values, _r_thr):
                    return _agg(_values, rel)
                if len(_values) == prev_size:
                    if rel == "=":
                        return None
                    elif rel in ["<", ">"]:
                        return _agg(_values, rel)
                    else:
                        raise ValueError(f"Relation not allowed: {rel}")

            return None

        values = np.asarray(values, dtype=np.float64)
        relations = np.asarray(relations, dtype=str)

        eq, gt, lt = _sort(values, relations)
        rels = set()

        if eq.shape[0] > 0:
            eq = _process(_values=eq, _r_thr=r_thr, _z_thr=z_thr, rel="=")
            if eq is not None:
                rels.add("=")
            else:
                return None, None
        if gt.shape[0] > 0:
            gt = _process(_values=gt, _r_thr=r_thr, _z_thr=z_thr, rel=">")
            if gt is not None:
                rels.add(">")
        if lt.shape[0] > 0:
            lt = _process(_values=lt, _r_thr=r_thr, _z_thr=z_thr, rel="<")
            if lt is not None:
                rels.add("<")

        if rels == {"="}:
            return eq, "="
        if rels == {">"}:
            return gt, ">"
        if rels == {"<"}:
            return lt, "<"

        if rels == {"=", ">"}:
            if eq >= gt or within_threshold(np.array([eq, gt]), r_thr):
                return eq, "="
            # eq < gt and outside threshold - contradiction; discard
        if rels == {"=", "<"}:
            if eq <= lt or within_threshold(np.array([eq, lt]), r_thr):
                return eq, "="
            # eq > lt and outside threshold - contradiction; discard
        if rels == {"<", ">"}:
            if gt < lt:
                if within_threshold(np.array([gt, lt]), r_thr):
                    return (lt + gt) / 2.0, "="
            if (gt - lt) < 0.5 * r_thr:
                return (lt + gt) / 2.0, "="

        if rels == {"=", "<", ">"}:
            gt_ok = eq >= gt or abs(gt - eq) <= r_thr
            lt_ok = eq <= lt or abs(eq - lt) <= r_thr
            if gt_ok and lt_ok:
                return eq, "="

        return None, None

    def process_group(sdf: pl.DataFrame) -> pl.DataFrame:
        nonlocal dd_col, smiles_col, value_col, relation_col, range_threshold, z_threshold

        dd_key = sdf[dd_col][0]
        smiles = sdf[smiles_col][0]
        values = sdf[value_col].to_numpy()
        relations = sdf[relation_col].to_numpy()

        val, rel = process_subset(
            values=values,
            relations=relations,
            r_thr=range_threshold,
            z_thr=z_threshold,
        )

        return pl.DataFrame(
            {
                dd_col: [dd_key],
                smiles_col: [smiles],
                value_col: [val],
                relation_col: [rel],
            },
            schema={
                dd_col: sdf.schema[dd_col],
                smiles_col: sdf.schema[smiles_col],
                value_col: pl.Float64,
                relation_col: pl.Utf8,
            },
        )

    int_df = df.select([dd_col, smiles_col, value_col, relation_col])

    rels = set(int_df[relation_col].unique())
    unexpected = rels - {"=", ">", "<"}
    if unexpected:
        raise ValueError(f"Relation not allowed: {unexpected}")

    mask = int_df[dd_col].is_duplicated()

    df_unique = int_df.filter(~mask)
    df_duplicated = int_df.filter(mask)

    dfs = []
    for dd_key in df_duplicated[dd_col].unique():
        sub_df = df_duplicated.filter(pl.col(dd_col) == dd_key)
        dfs.append(process_group(sub_df))

    if dfs:
        df_duplicated = pl.concat(dfs, how='vertical_relaxed')
        out_df = pl.concat([df_unique, df_duplicated], how='vertical_relaxed')
    else:
        out_df = df_unique

    df = df.drop([value_col, relation_col]).join(
        out_df.select([dd_col, value_col, relation_col]).unique(subset=dd_col),
        on=dd_col,
        how='left',
    )

    return df
