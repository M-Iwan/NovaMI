from typing import Union, List, Tuple, Optional, Iterable
import pickle
import os

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from rdkit import DataStructs


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

    quantiles = list(np.quantile(data, q=np.linspace(0, 1, n_bins+2)[1:-1]))

    def to_bin(value: float, qnts: List[float]):
        for idx, quantile in enumerate(qnts):
            if value < quantile:
                return idx + 1
        return len(quantiles) + 1

    bins = [to_bin(value, qnts=quantiles) for value in data]
    return bins


def combine_fingerprints(fps: Union[np.ndarray, List[np.ndarray]], method: str = 'mean',
                         decimals: int = 5):
    """
    Combine multiple fingerprints to use for Multi-Instance Learning approaches.

    Parameters
    ----------
    fps: Union[np.ndarray, List[np.ndarray]]
        An array or list of arrays with fingerprints
    method: str, optional
        How to combine the fingerprints:
        - 'mean'
        - 'sum'
        - 'max'
        - 'append'
        Default is 'mean'
    decimals: int, optional
        Number of decimals to keep. Default is 5

    Returns
    -------
    fps : np.ndarray
        Processed fingerprints
    """

    if isinstance(fps, np.ndarray):
        return np.round(fps, decimals)

    if isinstance(fps, list) & all(isinstance(fp, np.ndarray) for fp in fps):

        fps = np.stack(fps)

        # For selected methods, check if all arrays have the same length
        if method in ['mean', 'sum', 'max']:
            if method == 'mean':
                return np.round(np.mean(fps, axis=0), decimals)
            if method == 'sum':
                return np.round(np.sum(fps, axis=0), decimals)
            if method == 'max':
                return np.round(np.max(fps, axis=0), decimals)
        elif method == 'append':
            return np.round(fps.flatten(), decimals)
        else:
            raise ValueError('Possible options for < method > are: mean, sum, max, append')

    print(f'Expected < fps > to be np.ndarray or list, received < {type(fps)} > instead')
    return np.nan


def round_to_significant(x: Union[float, int], n: int):
    """
    Round a value X to N decimal places.

    Parameters
    ----------
    x: Union[float, int]
        The value to round
    n: int
        Number of significant digits to round to

    Returns
    -------
    float
    """

    if isinstance(x, str):
        return x
    if x is None:
        return 'None'

    if x == 0:
        return 0
    else:
        return round(x, n - int(np.floor(np.log10(abs(x)))) - 1)


def remove_nan_columns(array_list):
    """
    Remove all columns with at least one missing value

    Parameters
    ----------
    array_list: List[np.ndarray]
        List of arrays obtained from, for example, df[col].to_numpy()

    Returns
    -------
    array_list: List[np.ndarray]
        List of arrays with no missing values
    """
    array = np.vstack(array_list)

    mask = np.isnan(array).any(axis=0)
    array = array[:, ~mask]
    array_list = [array.reshape(-1) for array in np.vsplit(array, array.shape[0])]

    return array_list


def remove_inf_columns(array_list):
    array = np.vstack(array_list)

    mask = np.isinf(array).any(axis=0)
    array = array[:, ~mask]
    array_list = [array.reshape(-1) for array in np.vsplit(array, array.shape[0])]

    return array_list


def remove_zero_var_columns(array_list):
    array = np.vstack(array_list)

    mask = array.var(axis=0) == 0
    array = array[:, ~mask]
    array_list = [array.reshape(-1) for array in np.vsplit(array, array.shape[0])]

    return array_list


def remove_nan_df(df: pd.DataFrame, descriptor_col: Union[str, List[str]]):

    if isinstance(descriptor_col, str):
        descriptor_col = [descriptor_col]

    for desc in descriptor_col:
        array_list = df[desc].to_numpy()
        array_list = remove_nan_columns(array_list)
        df[desc] = array_list

    return df


def remove_inf_df(df: pd.DataFrame, descriptor_col: Union[str, List[str]]):

    if isinstance(descriptor_col, str):
        descriptor_col = [descriptor_col]

    for desc in descriptor_col:
        array_list = df[desc].to_numpy()
        array_list = remove_inf_columns(array_list)
        df[desc] = array_list

    return df


def remove_zero_var_df(df: pd.DataFrame, descriptor_col: Union[str, List[str]]):

    if isinstance(descriptor_col, str):
        descriptor_col = [descriptor_col]

    for desc in descriptor_col:
        array_list = df[desc].to_numpy()
        array_list = remove_zero_var_columns(array_list)
        df[desc] = array_list

    return df


def select_array(train_array: np.ndarray, eval_array: Optional[np.ndarray], eps: float = 1e-3,
                 save_dir: str = None, fold_idx: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray], VarianceThreshold]:
    """
    Remove columns with variance <= eps from passed arrays
    """

    selector = VarianceThreshold(threshold=eps)
    train_array = selector.fit_transform(train_array)

    if eval_array is not None:
        eval_array = selector.transform(eval_array)

    if save_dir is not None:
        selector_path = os.path.join(save_dir, f'selector_EFold_{fold_idx}.pkl')
        pickle.dump(selector, open(selector_path, 'wb'))

    return train_array, eval_array, selector


def scale_array(train_array: np.ndarray, eval_array: Optional[np.ndarray], rounded: Optional[int] = 7, scaler_method: str = 'iqr', save_dir: str = None,
                fold_idx: int = 0) -> Tuple[np.ndarray, Optional[
    np.ndarray], Union[StandardScaler, MinMaxScaler, RobustScaler]]:
    """
    Scale columns in numpy array using sklearn preprocessors
    """

    scalers = {
        'z_score': StandardScaler(),
        'min-max': MinMaxScaler(),
        'iqr': RobustScaler(unit_variance=True)
    }

    scaler = scalers.get(scaler_method)
    train_array = scaler.fit_transform(train_array)

    if eval_array is not None:
        eval_array = scaler.transform(eval_array)

    if rounded is not None:
        train_array = np.round(train_array, rounded)
        if eval_array is not None:
            eval_array = np.round(eval_array, rounded)

    if save_dir is not None:
        scaler_path = os.path.join(save_dir, f'scaler_EFold_{fold_idx}.pkl')
        pickle.dump(scaler, open(scaler_path, 'wb'))

    return train_array, eval_array, scaler


def standardize_array(train_array: np.ndarray, eval_array: Optional[np.ndarray], rounded: Optional[int],
                      scaler_method: str = 'iqr', eps: float = 1e-3, save_dir: str = None, fold_idx: int = 0):

    scalers = {
        'z_score': StandardScaler(),
        'min-max': MinMaxScaler(),
        'iqr': RobustScaler(unit_variance=True)
    }
    selector = VarianceThreshold(threshold=eps)
    scaler = scalers.get(scaler_method)

    train_array = selector.fit_transform(train_array)
    train_array = scaler.fit_transform(train_array)

    if eval_array is not None:
        eval_array = selector.transform(eval_array)
        eval_array = scaler.transform(eval_array)

    if rounded is not None:
        train_array = np.round(train_array, rounded)
        if eval_array is not None:
            eval_array = np.round(eval_array, rounded)

    if save_dir is not None:
        selector_path = os.path.join(save_dir, f'selector_EFold_{fold_idx}.pkl')
        scaler_path = os.path.join(save_dir, f'scaler_EFold_{fold_idx}.pkl')
        pickle.dump(selector, open(selector_path, 'wb'))
        pickle.dump(scaler, open(scaler_path, 'wb'))

    if eval_array is not None:
        return train_array, eval_array

    return train_array, None


def standardize_column_df(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame], in_col: str, out_col: str,
                          rounded: Optional[int], scaler: str = 'z-score', eps: float = 1e-3, save_dir: str = None):
    """
    Function for removing near-zero variance features and scaling data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Data for training
    test_df : pd.DataFrame | None
        Optional data for testing. If pd.DataFrame is passed the scaler will be fit using train_df and then transform
        also the test_df
    in_col: str
        Name of the column with values to scale. The expected format is 1D numpy ndarray in each row.
    out_col: str
        Name of the column for the output.
    rounded: int | None
        If not None: number of decimals to keep after scaling
    scaler: str
        How to scale the data:
        * z-score: scale the data to have mean of 0 and standard deviation of 1
        * min-max: scale the data to fall within < 0, 1 > range
        * iqr: scale the data using its median value and <0.25, 0.75> IQR
    eps: float
        Threshold for near-zero variance removal. Default is 1e-3
    save_dir: str | None
        If not None: path to a file where standardizer will be stored

    Returns
    -------
    train_df, test_df | None
    """

    if scaler == 'z-score':
        scaler = StandardScaler()
    elif scaler == 'min-max':
        scaler = MinMaxScaler()  # Usually used for k-NN
    elif scaler == 'iqr':
        scaler = RobustScaler(unit_variance=True)
    else:
        raise ValueError('Possible options for how are: z-score, min-max')

    train_df = train_df.copy()
    train_values = np.vstack(train_df[in_col].to_numpy())

    selector = VarianceThreshold(threshold=eps)
    train_values = selector.fit_transform(train_values)

    scaler.fit(train_values)
    transformed_train_values = scaler.transform(train_values)

    train_df.loc[:, out_col] = [array.reshape(-1) if array.size != 1 else array.reshape(-1).item()  # if an array holds only one element return it directly
                                for array in np.vsplit(transformed_train_values, train_values.shape[0])]

    if test_df is not None:
        test_df = test_df.copy()
        test_values = np.vstack(test_df[in_col].to_numpy())
        test_values = selector.transform(test_values)
        transformed_test_values = scaler.transform(test_values)
        test_df.loc[:, out_col] = [array.reshape(-1) if array.size != 1 else array.reshape(-1).item()
                                   for array in np.vsplit(transformed_test_values, test_values.shape[0])]

    if rounded is not None:
        train_df.loc[:, out_col] = train_df[out_col].apply(lambda array: np.round(array, rounded))
        if test_df is not None:
            test_df.loc[:, out_col] = test_df[out_col].apply(lambda array: np.round(array, rounded))

    if save_dir is not None:
        selector_path = save_dir.rstrip('/') + '/selector.pkl'
        scaler_path = save_dir.rstrip('/') + '/scaler.pkl'
        pickle.dump(selector, open(selector_path, 'wb'))
        pickle.dump(scaler, open(scaler_path, 'wb'))

    if test_df is not None:
        return train_df, test_df

    return train_df, None
