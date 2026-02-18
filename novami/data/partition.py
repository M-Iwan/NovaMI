import math
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold

from novami.data.manipulate import bin_data
from novami.data.cluster import butina_cluster, murcko_cluster, cc_cluster


def validate_dataframe(df: pl.DataFrame, features_col: Optional[str] = None, target_col: Optional[str] = None,
                       id_col: Optional[str] = None):

    """
    Validate the coherence of provided DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        A Polars DataFrame with data. Pandas DF will be internally cast to Polars DF.
    features_col: str
        Column holding molecular features encoded using 1D numpy arrays.
    target_col: str
        Column holding target values encoded using floats.
    id_col: str
        Column holding unique identifiers for each entry. If None a column named MolID will be added.

    Returns
    -------
    df: pl.DataFrame
    """

    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(f"Expected the DataFrame to be either Pandas or Polars DataFrame.\nGot {type(df)} instead.")
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if not all([isinstance(features_col, (str, None)), isinstance(target_col, (str, None))]):
        raise TypeError(f"Expected column names to be strings, got {type(features_col)} and {type(target_col)} instead.")

    if features_col is not None and features_col not in df.columns:
        raise KeyError(f"Features column <{features_col}> not found in DataFrame.")

    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Target column <{target_col}> not found in DataFrame.")

    if id_col is not None:
        if id_col not in df.columns:
            df = df.with_row_index(name='MolID')
        elif len(df) != len(df[id_col].unique()):
            raise ValueError(f"Column <{id_col}> not unique in DataFrame.")

    return df


def validate_features(x_array: np.ndarray):
    """
    Check validity of passed feature values and convert to 2D numpy array if needed.

    Parameters
    -----------
    x_array: np.ndarray
        Array of feature values to validate.

    Returns
    --------
    np.ndarray
        Validated feature array, guaranteed to be 2-dimensional.

    Raises
    -------
    TypeError
        If input is not a numpy.ndarray.
    ValueError
        If input has more than 2 dimensions.
    """

    if not isinstance(x_array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(x_array)} instead.")

    if x_array.ndim > 2:
        raise ValueError(f"Features may have at most 2 dimensions, got {x_array.ndim} dimensions instead.")

    if x_array.ndim == 1:  # a single entry
        x_array = x_array.reshape(1, -1)

    return x_array


def validate_targets(y_array: np.ndarray):
    """
    Check validity of passed target values and convert to 1D numpy array if needed.

    Parameters
    -----------
    y_array: np.ndarray
        Array of target values to validate.

    Returns
    --------
    np.ndarray
        Validated target array, guaranteed to be 1-dimensional.

    Raises
    -------
    TypeError
        If input is not a numpy.ndarray.
    ValueError
        If input has more than 2 dimensions or is a 2D array with more than 1 column.
    """

    if not isinstance(y_array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(y_array)} instead.")

    if (y_array.ndim == 2 and y_array.shape[1] > 1) or y_array.ndim > 2:
        raise ValueError(f"Features must be convertible to 1D array, got {y_array.ndim} dimensions instead.")

    if y_array.ndim == 2 and y_array.shape[1] == 1:
        y_array = y_array.reshape(-1)

    return y_array


def random_train_test_split(df: pl.DataFrame, fraction: float, id_col: str = 'MolID', seed: int = 42):
    """
    Randomly split the DataFrame into training and testing data.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with data.
    fraction: float
        Fraction of the data to use as TEST set.
    id_col: str
        Name of the column with unique identifier for each row. If missing, a "MolID" will be added.
    seed: int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_df, test_df: pl.DataFrame
    """

    if not isinstance(fraction, float):
        raise TypeError(f"Expected a float between 0 and 1, got {type(fraction)} instead.")

    if fraction < 0 or fraction > 1:
        raise ValueError(f"Expected a float between 0 and 1, got {fraction} instead.")

    df = validate_dataframe(df, id_col=id_col)

    test_df = df.sample(fraction=fraction, with_replacement=False, shuffle=True, seed=seed)
    train_df = df.filter(~pl.col(id_col).is_in(test_df[id_col].to_list()))

    return train_df, test_df


def stratified_train_test_split(df: pl.DataFrame, fraction: float, strat_col: Union[List[str], str],
                                id_col: str = 'MolID'):
    """
    Split the DataFrame into training and testing data while preserving the distributions denoted by strat_col.
    If the strat_col corresponds to a column of floats, they will first be binned based on the [0.2, 0.4, 0.6, 0.8]
    quantiles. This is deduced from the first item in the column. If multiple columns are provided, all data
    is combined in a single string.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with data.
    fraction: float
        Fraction of data to use for the TEST set.
    strat_col: List[str]
        List of columns to use for stratification.
    id_col: str
        Name of the column with unique identifier for each row. If missing, a "MolID" will be added.

    Returns
    -------
    train_df, test_df: pl.DataFrame
    """

    if not isinstance(fraction, float):
        raise TypeError(f"Expected a float between 0 and 1, got {type(fraction)} instead.")

    if fraction < 0 or fraction > 1:
        raise ValueError(f"Expected a float between 0 and 1, got {fraction} instead.")

    if not isinstance(strat_col, (str, list)):
        raise TypeError(f"Expected the < strat_col > to be a list or str, got {type(strat_col)} instead.")

    if isinstance(strat_col, str):
        strat_col = [strat_col]

    if id_col not in df.columns:
        df = df.with_row_index(name='MolID')

    for col in strat_col:
        if col not in df.columns:
            raise KeyError(f"Column <{col}> not found in DataFrame.")

    df = collate_strat(df=df, strat_col=strat_col)

    train_dfs = []
    test_dfs = []

    for value in df['Bin'].unique():
        sub_df = df.filter(pl.col('Bin') == value)
        train_df, test_df = random_train_test_split(sub_df, fraction=fraction)
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    train_df = pl.concat(train_dfs)
    test_df = pl.concat(test_dfs)

    return train_df, test_df


def minimal_train_test_split(df: pl.DataFrame, fraction: float, tolerance: float = 0.05, cluster_col: str = 'Cluster'):
    """
    Split the pre-defined clusters into training and testing data, assigning the smallest clusters to the test set.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame to split into folds.
    fraction: float
        Fraction of data to use for the TEST set.
    tolerance: float
        Allowed deviation in fraction.
    cluster_col: str
        Name of the column containing group identifiers.

    Returns
    -------
    train_df, test_df: pl.DataFrame
    """

    if not isinstance(fraction, float):
        raise TypeError(f"Expected a float between 0 and 1, got {type(fraction)} instead.")

    if fraction < 0 or fraction > 1:
        raise ValueError(f"Expected a float between 0 and 1, got {fraction} instead.")

    if not isinstance(cluster_col, str):
        raise TypeError(f"Expected the < cluster_col > to be str, got {type(cluster_col)} instead.")

    cluster_counts = df[cluster_col].value_counts(name='Count').sort('Count')
    n_test = math.ceil(len(df) * fraction)
    top_test = math.ceil(len(df) * (fraction + tolerance))

    clusters = []  # These are test clusters
    test_size = 0

    for cluster, cluster_size in cluster_counts.iter_rows():
        if test_size < n_test:
            clusters.append(cluster)
            test_size += cluster_size
        else:
            break

    if test_size > top_test:
        print(f'Unable to split data into train and test. Fraction: {fraction}. Tolerance: {tolerance}.')
        return None, None

    df = df.with_columns(
        pl.when(pl.col(cluster_col).is_in(clusters)).then(pl.lit("Test")).otherwise(pl.lit("Train")).alias("Set")
    )

    return df


def random_kfold_split(df: pl.DataFrame, n_folds: int = 5, seed: int = 42):
    """
    Assigns k-fold cross-validation fold indices to rows in a polars DataFrame.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame to split into folds.
    n_folds: int, default=5
        Number of folds for k-fold cross-validation.
    seed: int, default=42
        Random seed for reproducibility.

    Returns
    --------
    pl.DataFrame
        Input DataFrame with an additional 'Fold' column containing fold indices (0 to n_folds-1) for each row.
    """

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    df = df.with_columns(pl.lit(-1).alias('Fold'))
    x_array = np.zeros(len(df))

    for idx, (_, test_idx) in enumerate(kf.split(x_array)):
        df[test_idx, 'Fold'] = idx

    return df


def stratified_kfold_split(df: pl.DataFrame, strat_col: str, n_folds: int = 5, seed: int = 42):
    """
    Assigns stratified k-fold cross-validation fold indices to rows in a polars DataFrame.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame to split into stratified folds.
    strat_col: str
        Name of the column to use for stratification.
    n_folds: int, default=5
        Number of folds for stratified k-fold cross-validation.
    seed: int, default=42
        Random seed for reproducibility.

    Returns
    --------
    pl.DataFrame
        Input DataFrame with an additional 'Fold' column containing stratified fold indices.
    """

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = collate_strat(df=df, strat_col=strat_col).with_columns(pl.lit(-1).alias('Fold'))

    x_array = np.zeros(len(df))
    y_array = df['Bin'].to_numpy()

    for idx, (_, test_idx) in enumerate(skf.split(x_array, y_array)):
        df[test_idx, 'Fold' ] = idx

    return df


def group_kfold_split(df: pl.DataFrame, cluster_col: str, n_folds: int = 5, seed: int = 42):
    """
    Assigns group-based k-fold cross-validation fold indices to rows in a polars DataFrame.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame to split into group-based folds.
    cluster_col: str
        Name of the column containing group identifiers.
    n_folds: int, default=5
        Number of folds for group k-fold cross-validation.
    seed: int, default=42
        Random seed for reproducibility.

    Returns
    --------
    pl.DataFrame
        Input DataFrame with an additional 'Fold' column containing group-based fold indices.
    """

    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = df.with_columns(pl.lit(-1).alias('Fold'))

    x_array = np.zeros(len(df))
    y_array = np.zeros(len(df))

    for idx, (_, test_idx) in enumerate(gkf.split(x_array, y_array, groups=df[cluster_col])):
        df[test_idx, 'Fold' ] = idx

    return df


def stratified_group_kfold_split(df: pl.DataFrame, strat_col: Union[str, List[str]], cluster_col: str, n_folds: int = 5,
                                 seed: int = 42):
    """
    Assigns stratified group-based k-fold cross-validation fold indices to rows in a polars DataFrame.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame to split into stratified group-based folds.
    strat_col: Union[str, List[str]]
        Name of the column or list of columns to use for stratification.
    cluster_col: str
        Name of the column containing group identifiers.
    n_folds: int, default=5
        Number of folds for stratified group k-fold cross-validation.
    seed: int, default=42
        Random seed for reproducibility.

    Returns
    --------
    pl.DataFrame
        Input DataFrame with an additional 'Fold' column containing stratified group-based fold indices.
    """

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = collate_strat(df=df, strat_col=strat_col).with_columns(pl.lit(-1).alias('Fold'))

    x_array = np.zeros(len(df))
    y_array = df['Bin'].to_numpy()
    groups = df[cluster_col].to_numpy()

    for idx, (_, test_idx) in enumerate(sgkf.split(x_array, y_array, groups=groups)):
        df[test_idx, 'Fold' ] = idx

    return df


def temporal_split(df: pl.DataFrame, time_col: str, fraction: float = 0.2):
    """
    Splits a DataFrame into training and testing sets based on temporal ordering.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame to split.
    time_col: str
        Name of the column containing temporal values for ordering.
    fraction: float, default=0.2
        Approximate fraction of data to allocate to the test set (most recent data).

    Returns
    --------
    df: pl.DataFrame
    """

    n_samples = int(round(len(df) * fraction))

    df = df.sort(by=time_col, descending=True)
    time_value = df[time_col].item(n_samples)

    df = df.with_columns(
        pl.when(pl.col(time_col) < time_value).then(pl.lit("Train")).otherwise(pl.lit("Test")).alias("Set")
    )

    return df


def murcko_kfold_split(df: pl.DataFrame, strat_col: Union[str, List[str]] = None, smiles_col: str = 'SMILES',
                       n_folds: int = 5, generic: bool = False, tolerance: float = 0.3):
    """
    Splits a DataFrame into folds based on Murcko scaffolds.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame containing chemical structures.
    strat_col: Union[str, List[str]], optional
        Name of the column or list of columns to use for stratification. If None, no stratification is performed.
    smiles_col: str, default='SMILES'
        Name of the column containing SMILES representations of molecules.
    n_folds: int, default=5
        Number of folds for cross-validation.
    generic: bool, default=False
        Whether to use generic Murcko scaffolds (without atom types) or specific scaffolds.
    tolerance: float, default=0.3
        Maximum acceptable Relative Standard Deviation between fold sizes.

    Returns
    --------
    pl.DataFrame
        Input DataFrame with additional 'Cluster' and 'Fold' columns containing scaffold cluster IDs
        and fold indices.
    """

    unique_df = df.unique(subset=smiles_col, keep='first')
    unique_df = murcko_cluster(df=unique_df, smiles_col=smiles_col, generic=generic)

    df = df.join(unique_df, how='inner', on=smiles_col)

    if strat_col is not None:
        df = stratified_group_kfold_split(df=df, strat_col=strat_col, cluster_col='Cluster', n_folds=n_folds)
    else:
        df = group_kfold_split(df=df, cluster_col='Cluster', n_folds=n_folds)

    fold_sizes = df['Fold'].value_counts(name='Size')['Size'].to_numpy()

    if not are_folds_balanced(fold_sizes, tolerance=tolerance):
        raise RuntimeError(f'Folds are not balanced at < {tolerance} > tolerance')

    return df


def butina_kfold_split(df: pl.DataFrame, strat_col: Union[str, List[str]] = None, smiles_col: str = 'SMILES',
                       fp_col: str = 'Morgan', threshold: float = 0.4, batch_size: int = 512, n_folds: int = 5,
                       n_jobs: int = 1, tolerance: float = 0.3):

    """
    Splits a DataFrame into folds based on Butina clustering.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame containing chemical structures.
    strat_col: Union[str, List[str]], optional
        Name of the column or list of columns to use for stratification. If None, no stratification is performed.
    smiles_col: str, default='SMILES'
        Name of the column containing SMILES representations of molecules.
    fp_col: str, default='Morgan'
        Name of the column containing molecular fingerprints for distance calculation.
    threshold: float, default=0.4
        Distance threshold for Butina clustering.
    batch_size: int, default=512
        Number of molecules to process in each batch.
    n_folds: int, default=5
        Number of folds for cross-validation.
    n_jobs: int, default=11
        Number of parallel jobs to run (-1 means using all processors).
    tolerance: float, default=0.3
        Maximum acceptable Relative Standard Deviation between fold sizes.

    Returns
    --------
    pl.DataFrame
        Input DataFrame with additional 'Cluster' and 'Fold' columns containing cluster IDs
        and fold indices.
    """

    unique_df = df.unique(subset=smiles_col, keep='first')
    unique_df = butina_cluster(df=unique_df, fp_col=fp_col, threshold=threshold, batch_size=batch_size, n_jobs=n_jobs)

    df = df.join(unique_df[[smiles_col, 'Cluster']], how='inner', on=smiles_col)

    if strat_col is not None:
        df = stratified_group_kfold_split(df=df, strat_col=strat_col, cluster_col='Cluster', n_folds=n_folds)
    else:
        df = group_kfold_split(df=df, cluster_col='Cluster', n_folds=n_folds)

    fold_sizes = df['Fold'].value_counts(name='Size')['Size'].to_numpy()

    if not are_folds_balanced(fold_sizes, tolerance=tolerance):
        raise RuntimeError(f'Folds are not balanced at < {tolerance} > tolerance and < {threshold} > threshold.')

    return df


def cc_kfold_split(df: pl.DataFrame, strat_col: Union[str, List[str]] = None, smiles_col: str = 'SMILES',
                   features_col: str = 'Morgan', threshold: float = 0.3, metric: str = 'jaccard', n_folds: int = 5,
                   n_jobs: int = 11, tolerance: float = 0.3):

    """
    Splits a DataFrame into folds based on connected components clustering.

    Parameters
    -----------
    df: pl.DataFrame
        Input DataFrame containing chemical structures.
    strat_col: Union[str, List[str]], optional
        Name of the column or list of columns to use for stratification. If None, no stratification is performed.
    smiles_col: str, default='SMILES'
        Name of the column containing SMILES representations of molecules.
    features_col: str, default='Morgan'
        Name of the column containing molecular fingerprints for distance calculation.
    threshold: float, default=0.3
        Distance threshold for connected components clustering.
    metric: str, default='jaccard'
        Distance metric to use for clustering.
    n_folds: int, default=5
        Number of folds for cross-validation.
    n_jobs: int, default=1
        Number of parallel jobs to run (-1 means using all processors).
    tolerance: float, default=0.3
        Maximum acceptable Relative Standard Deviation between fold sizes.

    Returns
    --------
    pl.DataFrame
        Input DataFrame with additional 'Cluster' and 'Fold' columns containing cluster IDs
        and fold indices.
    """

    unique_df = df.unique(subset=smiles_col, keep='first')
    unique_df = cc_cluster(unique_df, features_col=features_col, threshold=threshold, metric=metric, n_jobs=n_jobs)

    df = df.join(unique_df[[smiles_col, 'Cluster']], how='inner', on=smiles_col)

    if strat_col is not None:
        df = stratified_group_kfold_split(df=df, strat_col=strat_col, cluster_col='Cluster', n_folds=n_folds)
    else:
        df = group_kfold_split(df=df, cluster_col='Cluster', n_folds=n_folds)

    fold_sizes = df['Fold'].value_counts(name='Size')['Size'].to_numpy()

    if not are_folds_balanced(fold_sizes, tolerance=tolerance):
        raise RuntimeError(f'Folds are not balanced at < {tolerance} > tolerance and < {threshold} > threshold.')

    return df


def collate_strat(df: pl.DataFrame, strat_col: Union[str, List[str]]):
    """
    Combine multiple columns to form a new stratification key. String columns are used as-is, integer columns
    are converted to strings, while float columns are binned (5 bins) and then converted to strings.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    strat_col: Union[str, List[str]]
        Column(s) to be combined

    Returns
    -------
    df: pl.DataFrame
    """

    def cast_column(data: pl.Series):
        col_type = type(data.item(0))
        if col_type is str:
            return data
        elif col_type is int:
            return data.cast(pl.String)
        elif col_type is float:
            return bin_data(data.to_list(), n_bins=5)
        else:
            raise TypeError(f"Expected column to be one of <int, float, str>, got {col_type} instead.")

    strat_df = pl.DataFrame({
        f'Bin_{idx}': cast_column(df[col]) for idx, col in enumerate(strat_col)
    })

    df = pl.concat([
        df,
        strat_df.with_columns(
            pl.concat_str([pl.col(col) for col in strat_col], separator=':').alias('Bin')
        )[['Bin']]
    ])

    return df


def are_folds_balanced(fold_sizes: np.ndarray, tolerance: float = 0.3):
    """
    Verify if fold sizes are balanced based on relative standard deviation.

    Parameters
    ----------
    fold_sizes: np.ndarray
        A numpy array of integers
    tolerance: float, default = 0.3
        Maximum acceptable relative standard deviation between fold sizes.

    Returns
    -------
    bool
    """
    if any(fold_sizes < 0):
        raise ValueError(f"Fold sizes should be non-negative, got {fold_sizes}")

    if (mean := fold_sizes.mean()) == 0:
        raise ValueError(f"Empty folds passed: {fold_sizes}")

    std = np.std(fold_sizes, ddof=1)

    rsd = std/mean
    return rsd <= tolerance
