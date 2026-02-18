from typing import Optional, Dict

import numpy as np
import pandas as pd
import polars as pl


class KFoldManager:
    """
    Manages splitting data based on pre-defined folds.

    Parameters
    ----------
    df : pl.DataFrame
        The stored DataFrame containing all data.
    smiles_col : str
        Name of the column containing SMILES strings.
    features_col : str
        Name of the column containing features.
    target_col : str
        Name of the column containing target values.
    fold_col : str
        Name of the column containing fold assignments.
    test_fold : int
        Fold to be used for testing.
    weights_col : str or None
        Name of the column containing sample weights, if provided.
    groups_col : str or None
        Name of the column containing groups, if provided.

    Attributes
    ----------
    df : pl.DataFrame
        The stored DataFrame containing all data.
    smiles_col : str
        Name of the column containing SMILES strings.
    features_col : str
        Name of the column containing features.
    target_col : str
        Name of the column containing target values.
    fold_col : str
        Name of the column containing fold assignments.
    test_fold : int
        Fold to be used for testing.
    weights_col : str or None
        Name of the column containing sample weights, if provided.
    groups_col : str or None
        Name of the column containing groups, if provided.

    x_array : numpy.ndarray
        2D array of feature values extracted from the DataFrame.
    y_true : numpy.ndarray
        1D array of target values extracted from the DataFrame.
    w_array : numpy.ndarray
        1D array of sample weights (ones if weights_col is None).
    g_array : numpy.ndarray
        1D array of group assignments.
    s_array : numpy.ndarray
        1D array of SMILES strings.
    splits : numpy.ndarray
        Fold assignments for each sample.
    folds : numpy.ndarray
        Unique fold identifiers. Test fold is excluded.
    train_idxs : dict
        Dictionary mapping each fold to indices for training data (all samples not in that fold).
    eval_idxs: dict
        Dictionary mapping each fold to indices for evaluation data (samples in that fold).
    test_idxs: np.ndarray
        Row indices for test samples.
    non_test_idxs: np.ndarray
        Row indices for non-test samples.

    Methods
    -------
    get_train_data(fold)
        Get training data (features, targets, weights) for all folds except the specified one.
    get_eval_data(fold)
        Get evaluation data (features, targets, weights, groups) for the specified fold.
    get_test_data()
        Get testing data (features, targets, weights, groups) for the test fold.
    get_non_test_data()
        Get non-test data (features, targets, weights).
    """
    def __init__(self, df: pl.DataFrame, smiles_col: str, features_col: str, target_col: str, fold_col: str, test_fold: int,
                 weights_col: Optional[str] = None, groups_col: Optional[str] = None):

        required_cols = [smiles_col, features_col, target_col, fold_col]
        if weights_col is not None:
            required_cols.append(weights_col)
        if groups_col is not None:
            required_cols.append(groups_col)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f'Missing column(s) in DataFrame: {missing_cols}')

        self.df = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
        self.smiles_col = smiles_col
        self.features_col = features_col
        self.target_col = target_col
        self.fold_col = fold_col
        self.test_fold = test_fold
        self.weights_col = weights_col
        self.groups_col = groups_col

        self.x_array = np.vstack(self.df[self.features_col].to_numpy())  # 2D numpy array
        self.y_true = self.df[self.target_col].to_numpy()  # 1D numpy array
        self.s_array = self.df[self.smiles_col].to_numpy() # 1D numpy array

        if self.weights_col is not None:
            self.w_array = self.df[self.weights_col].to_numpy() # 1D numpy array
        else:
            self.w_array = None

        if self.groups_col is not None:
            self.g_array = self.df[self.groups_col].to_numpy() # 1D numpy array of whatever
        else:
            self.g_array = None

        self.splits = self.df[self.fold_col].to_numpy()
        all_folds = np.unique(self.splits)

        self.folds = np.array([fold for fold in all_folds if fold != self.test_fold])

        self.train_idxs = {fold: np.where((self.splits != fold) & (self.splits != self.test_fold))[0] for fold in self.folds}
        self.eval_idxs = {fold: np.where(self.splits == fold)[0] for fold in self.folds}
        self.test_idxs = np.where(self.splits == self.test_fold)[0]
        self.non_test_idxs = np.where(self.splits != self.test_fold)[0]

    def get_train_data(self, fold: int) -> Dict[str, np.ndarray]:
        if fold == self.test_fold:
            raise ValueError(f'Cannot access test fold < {self.test_fold} > during training')

        if fold not in self.folds:
            raise ValueError(f'Fold {fold} is not in present in DataFrame')

        idx = self.train_idxs[fold]
        return {
            'x_array': self.x_array[idx, :],
            'y_true': self.y_true[idx],
            'sample_weight': self.w_array[idx] if self.w_array is not None else None,
            'groups': self.g_array[idx] if self.g_array is not None else None
        }

    def get_eval_data(self, fold: int) -> Dict[str, np.ndarray]:
        if fold == self.test_fold:
            raise ValueError(f'Cannot access test fold < {self.test_fold} > during evaluation')
        if fold not in self.folds:
            raise ValueError(f'Fold {fold} is not in present in DataFrame')

        idx = self.eval_idxs[fold]
        return {
            'x_array': self.x_array[idx, :],
            'y_true': self.y_true[idx],
            'sample_weight': self.w_array[idx] if self.w_array is not None else None,
            'groups': self.g_array[idx] if self.g_array is not None else None
        }

    def get_test_data(self) -> Dict[str, np.ndarray]:
        return {
            'x_array': self.x_array[self.test_idxs, :],
            'y_true': self.y_true[self.test_idxs],
            'sample_weight': self.w_array[self.test_idxs] if self.w_array is not None else None,
            'groups': self.g_array[self.test_idxs] if self.g_array is not None else None
        }

    def get_non_test_data(self) -> Dict[str, np.ndarray]:
        return {
            'x_array': self.x_array[self.non_test_idxs, :],
            'y_true': self.y_true[self.non_test_idxs],
            'sample_weight': self.w_array[self.non_test_idxs] if self.w_array is not None else None,
            'groups': self.g_array[self.non_test_idxs] if self.g_array is not None else None
        }

    def get_train_smiles(self, fold: int):
        return self.s_array[self.train_idxs[fold]]

    def get_eval_smiles(self, fold: int):
        return self.s_array[self.eval_idxs[fold]]

    def get_test_smiles(self):
        return self.s_array[self.test_idxs]

    def get_non_test_smiles(self):
        return self.s_array[self.non_test_idxs]


class TTManager:
    """
    TrainTestManager for data manipulation based on train/test splits.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing feature data, target values, and set assignments.
    smiles_col : str
        Name of the column containing SMILES strings.
    features_col : str
        Name of the column containing features.
    target_col : str
        Name of the column containing target values.
    set_col : str
        Name of the column containing set assignments.
    weights_col : str or None
        Name of the column containing sample weights, if provided.
    groups_col : str or None
        Name of the column containing groups, if provided.


    Attributes
    ----------
    df : pl.DataFrame
        Polars DataFrame containing feature data, target values, and set assignments.
    smiles_col : str
        Name of the column containing SMILES strings.
    features_col : str
        Name of the column containing features.
    target_col : str
        Name of the column containing target values.
    set_col : str
        Name of the column containing set assignments.
    weights_col : str or None
        Name of the column containing sample weights, if provided.
    groups_col : str or None
        Name of the column containing groups, if provided.

    x_array : numpy.ndarray
        2D array of feature values extracted from the DataFrame.
    y_true : numpy.ndarray
        1D array of target values extracted from the DataFrame.
    w_array : numpy.ndarray
        1D array of sample weights (ones if weights_col is None).
    g_array : numpy.ndarray
        1D array of group assignments.
    s_array : numpy.ndarray
        1D array of SMILES strings.
    train_idxs : dict
        Row indices for train samples
    test_idxs: np.ndarray
        Row indices for test samples.

    Methods
    -------
    get_train_data()
        Get training data (features, targets, weights) for the training set.
    get_test_data()
        Get testing data (features, targets, weights, groups) for the test set.
    """
    def __init__(self, df: pl.DataFrame, smiles_col: str, features_col: str, target_col: str, set_col: str,
                 weights_col: Optional[str] = None, groups_col: Optional[str] = None):

        required_cols = [smiles_col, features_col, target_col]
        if weights_col is not None:
            required_cols.append(weights_col)
        if groups_col is not None:
            required_cols.append(groups_col)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f'Missing column(s) in DataFrame: {missing_cols}')

        self.df = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df

        self.smiles_col = smiles_col
        self.features_col = features_col
        self.target_col = target_col
        self.set_col = set_col
        self.weights_col = weights_col
        self.groups_col = groups_col

        self.x_array = np.vstack(self.df[self.features_col].to_numpy())  # 2D numpy array
        self.y_true = self.df[self.target_col].to_numpy()  # 1D numpy array
        self.s_array = self.df[self.smiles_col].to_numpy() # 1D numpy array

        if self.weights_col is not None:
            self.w_array = self.df[self.weights_col].to_numpy() # 1D numpy array
        else:
            self.w_array = None

        if self.groups_col is not None:
            self.g_array = self.df[self.groups_col].to_numpy() # 1D numpy array of whatever
        else:
            self.g_array = None

        self.splits = self.df[self.set_col].to_numpy()
        self.train_idxs = np.where(self.splits == "Train")[0]
        self.test_idxs = np.where(self.splits == "Test")[0]

    def get_train_data(self) -> Dict[str, np.ndarray]:
        return {
            'x_array': self.x_array[self.train_idxs, :],
            'y_true': self.y_true[self.train_idxs],
            'sample_weight': self.w_array[self.train_idxs] if self.w_array is not None else None,
            'groups': self.g_array[self.train_idxs] if self.g_array is not None else None
        }

    def get_test_data(self) -> Dict[str, np.ndarray]:
        return {
            'x_array': self.x_array[self.test_idxs, :],
            'y_true': self.y_true[self.test_idxs],
            'sample_weight': self.w_array[self.test_idxs] if self.w_array is not None else None,
            'groups': self.g_array[self.test_idxs] if self.g_array is not None else None
        }

    def get_train_smiles(self):
        return self.s_array[self.train_idxs]

    def get_test_smiles(self):
        return self.s_array[self.test_idxs]