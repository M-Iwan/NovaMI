"""Dataset evaluation script for Snellius"""

import argparse
import pandas as pd
import numpy as np
import os
import pickle
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from typing import List, Dict, Union, ClassVar, Tuple, Optional
from collections import defaultdict
from copy import deepcopy
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.special import expit
import optuna


def merge_columns(row, cols: List[str]) -> np.ndarray:
    """
    Combine arrays from multiple columns into a single numpy array
    """
    array = np.concatenate([row[col] for col in cols])
    return np.round(array.reshape(-1), 7)


def get_model(name: str) -> Union[LogisticRegression, RandomForestClassifier, SVC, XGBClassifier]:
    """
    Dictionary of name to model
    """
    models = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "XGBClassifier": XGBClassifier
    }

    if name not in models.keys():
        raise ValueError(f'Unknown model: {name}')
    return models.get(name)


def remove_nan_columns(array_list: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    """
    Remove all columns with at least 1 missing value
    """
    array = np.vstack(array_list)

    mask = np.isnan(array).any(axis=0)
    array = array[:, ~mask]
    array_list = [array.reshape(-1) for array in np.vsplit(array, array.shape[0])]

    return array_list


def remove_inf_columns(array_list: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    """
    Remove all columns with at least 1 infinite value
    """
    array = np.vstack(array_list)

    mask = np.isinf(array).any(axis=0)
    array = array[:, ~mask]
    array_list = [array.reshape(-1) for array in np.vsplit(array, array.shape[0])]

    return array_list


def remove_zero_var_columns(array_list: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    """
    Remove all columns with zero variance
    """
    array = np.vstack(array_list)

    mask = array.var(axis=0) == 0
    array = array[:, ~mask]
    array_list = [array.reshape(-1) for array in np.vsplit(array, array.shape[0])]

    return array_list


def remove_nan_df(df: pd.DataFrame, descriptor_col: Union[str, List[str]]) -> pd.DataFrame:
    """
    Iterate over columns in dataframe and remove columns with at least 1 missing value from underlying numpy arrays
    """
    if isinstance(descriptor_col, str):
        descriptor_col = [descriptor_col]

    for desc in descriptor_col:
        array_list = df[desc].to_numpy()
        array_list = remove_nan_columns(array_list)
        df[desc] = array_list

    return df


def remove_inf_df(df: pd.DataFrame, descriptor_col: Union[str, List[str]]) -> pd.DataFrame:
    """
    Iterate over columns in dataframe and remove columns with at least 1 infinite value from underlying numpy arrays
    """
    if isinstance(descriptor_col, str):
        descriptor_col = [descriptor_col]

    for desc in descriptor_col:
        array_list = df[desc].to_numpy()
        array_list = remove_inf_columns(array_list)
        df[desc] = array_list

    return df


def remove_zero_var_df(df: pd.DataFrame, descriptor_col: Union[str, List[str]]) -> pd.DataFrame:
    """
    Iterate over columns in dataframe and remove columns with zero variance from underlying numpy arrays
    """
    if isinstance(descriptor_col, str):
        descriptor_col = [descriptor_col]

    for desc in descriptor_col:
        array_list = df[desc].to_numpy()
        array_list = remove_zero_var_columns(array_list)
        df[desc] = array_list

    return df


class IndexedDict(dict):
    def __getitem__(self, key):
        if isinstance(key, (int, str)):
            # Regular dict behavior for single int or string key
            return super().__getitem__(key)
        elif isinstance(key, (np.ndarray, list)):
            sliced = {}
            for k, arr in self.items():
                if isinstance(arr, np.ndarray):
                    sliced[k] = arr[key]
                else:
                    sliced[k] = arr
            return sliced
        else:
            raise ValueError(f'Unrecognized key format: {type(key)}')


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


def get_params_xgb_classifier(trial) -> Dict:
    """
    Suggest values for XGBClassifier for optuna
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),
        'max_leaves': trial.suggest_int('max_leaves', 0, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }


def get_params_random_forest_classifier(trial) -> Dict:
    """
    Suggest values for RandomForestClassifier for optuna
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 0.05, log=True)
    }


def get_params_svc(trial) -> Dict:
    """
    Suggest values for SVC for optuna
    """
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel
    }
    if kernel in ['poly', 'rbf', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-3, 1.0, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    if kernel in ['poly', 'sigmoid']:
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

    return params


def get_params_logistic_regression(trial) -> Dict:
    """
    Suggest values for LogisticRegression for optuna
    """
    valid_combinations = [
        'lbfgs:l2', 'lbfgs:None',
        'liblinear:l1', 'liblinear:l2',
        'newton-cg:l2', 'newton-cg:None',
        'newton-cholesky:l2', 'newton-cholesky:None',
        'sag:l2', 'sag:None', 'saga:elasticnet',
        'saga:l1', 'saga:l2', 'saga:None'
    ]
    combination = trial.suggest_categorical('solver_penalty', valid_combinations)
    solver, penalty = combination.split(':')
    penalty = None if penalty == 'None' else penalty

    params = {
        'solver': solver,
        'penalty': penalty,
        'C': trial.suggest_float('C', 0.001, 10, log=True),
        'max_iter': 1024,
    }

    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    return params


def get_params_knn_classifier(trial) -> Dict:
    """
    Suggest values for KNeighborsClassifier for optuna
    """
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 25),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
        'p': trial.suggest_int('p', 1, 2)
    }

    return params


def inner_score(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, y_wgts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Used as inner function with 1D numpy arrays.

    Parameters
    ----------

    y_true: np.ndarray
        Array of true labels
    y_pred: np.ndarray
        Array of predicted labels.
    y_score: np.ndarray
        Array of predicted probabilities (from .predict_proba or .decision_function)
    y_wgts: np.ndarray
        Array of weights. Optional.
    """

    def safe_div(numerator, denominator, default_=0.0):
        return numerator / denominator if denominator != 0 else default_

    if y_wgts is None:  # we don't know the weights
        y_wgts = np.ones_like(y_true)  # just say they are equal and don't think about it anymore!

    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts)

    if len(np.unique(y_true)) != 2:
        return {}

    tn, fp, fn, tp = conf_mat.ravel()

    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)

    metrics = {
        'TP': np.round(tp, 5),
        'FP': np.round(fp, 5),
        'FN': np.round(fn, 5),
        'TN': np.round(tn, 5),
        'Accuracy': np.round(safe_div(tp + tn, tp + fp + fn + tn), 5),
        'Recall': np.round(rec, 5),
        'Specificity': np.round(spec, 5),
        'Precision': np.round(safe_div(tp, tp + fp), 5),
        'Balanced Accuracy': np.round((rec + spec) / 2, 5),
        'GeomRS': np.round(np.sqrt(rec * spec), 5),
        'HarmRS': np.round(safe_div(2 * rec * spec, rec + spec), 5),
        'F1 Score': np.round(safe_div(2 * tp, 2 * tp + fp + fn), 5),
        'ROC AUC': np.round(roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=y_wgts), 5),
        'MCC': np.round(safe_div((tp * tn) - (fp * fn), np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), 5)
    }

    return metrics


def outer_score(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, y_wgts: np.ndarray, y_sign: dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    Aggregate predictions based on different criteria. Intended to be used
    with classical ML metrics, purely on predictions

    Parameters
    ----------

    y_true: np.ndarray
        Array of true labels.
    y_pred: np.ndarray
        Array of predicted labels.
    y_score: np.ndarray
        Array of predicted probabilities (from .predict_proba or .decision_function)
    y_wgts: np.ndarray
        Array of weights. Optional.
    y_sign: IndexedDict
        Custom dictionary in form key: np.array(List[str]), where np arrays can be used to slice it.
    """

    def convert_array(array):
        if array.size == 0:
            raise ValueError(f'Received an empty array')
        elif len(array.shape) == 1:
            return array
        elif array.shape[1] == 1:
            return array.flatten()
        else:
            raise ValueError(f'Unrecognized array shape: {array.shape}')

    y_true = convert_array(y_true)
    y_pred = convert_array(y_pred)
    y_score = convert_array(y_score)
    y_wgts = convert_array(y_wgts)
    y_sign = IndexedDict({key: convert_array(item) for key, item in y_sign.items()})

    # Calculate Overall metrics
    total_metrics = inner_score(y_true=y_true, y_pred=y_pred, y_score=y_score, y_wgts=y_wgts)

    # Calculate per-sign metrics
    per_sign_metrics = defaultdict(dict)  # i.e. sign_type: task: sign_name: metrics | horrible
    for sign_type, sign_values in y_sign.items():
        unique_signs = set(sign_values)
        for sign_name in unique_signs:
            key = (sign_type, sign_name)
            sign_idx = np.where(sign_values == sign_name)[0]
            if len(sign_idx) > 0:
                per_sign_metrics[key] = inner_score(
                    y_true=y_true[sign_idx],
                    y_pred=y_pred[sign_idx],
                    y_score=y_score[sign_idx],
                    y_wgts=y_wgts[sign_idx]
                )
            else:
                per_sign_metrics[key] = {}

    return {
        'Total': total_metrics,
        'Sign': per_sign_metrics
    }


class FoldUnit:
    """
    FoldUnit encapsulates training, evaluation, and transformation logic for a single fold
    during cross-validation.

    This class manages model fitting, prediction, and scoring using fold-specific feature
    selectors and scalers. It assumes external assignment of selectors and scalers
    through class variables.

    Attributes
    ----------
    selectors : ClassVar[Dict[int, VarianceThreshold]]
        Dictionary mapping fold index to a VarianceThreshold feature selector.
    scalers : ClassVar[Dict[int, RobustScaler]]
        Dictionary mapping fold index to a RobustScaler for normalization.
    train_idxs : ClassVar[Dict[int, np.ndarray]]
        Mapping of fold index to training sample indices.
    eval_idxs : ClassVar[Dict[int, np.ndarray]]
        Mapping of fold index to evaluation sample indices.
    model : object
        A scikit-learn-compatible model instance used for training and inference.
    fold : int
        The fold index corresponding to this unit.
    scores : dict
        A dictionary storing evaluation metrics across training and evaluation sets.

    Methods
    -------
    fit(x_train, x_demo, y_array, y_wgts)
        Fits the model on transformed training data.
    predict(x_array, x_demo)
        Predicts class labels for the given input.
    predict_proba(x_array, x_demo)
        Predicts probabilities (or scores) for the positive class.
    transform(array)
        Applies the fold-specific selector and scaler to the input features.
    """

    selectors:  ClassVar[Dict[int, VarianceThreshold]] = {}
    scalers:    ClassVar[Dict[int, RobustScaler]] = {}
    train_idxs: ClassVar[Dict[int, np.ndarray]] = {}
    eval_idxs:  ClassVar[Dict[int, np.ndarray]] = {}

    def __init__(self, model, fold_idx: int):
        self.model = model
        self.fold = fold_idx
        self.scores = {}

    def __repr__(self):
        return f"<FoldUnit(fold={self.fold}, model={self.model.__class__.__name__})>"

    def __str__(self):
        eval_score = self.scores.get(('Eval', 'Weighted'), {})
        total_metrics = eval_score.get('Total', {})
        metrics_str = ', '.join(f"{k}: {v:.4f}" for k, v in total_metrics.items())
        return f"Fold {self.fold} | Model: {self.model.__class__.__name__} | Eval Scores: {metrics_str}"

    def fit(self, x_train: np.ndarray, x_demo: np.ndarray, y_array: np.ndarray, y_wgts: np.ndarray):
        x_array = self.transform(x_train)
        x_array = np.hstack((x_array, x_demo))
        self.model.fit(x_array, y_array, sample_weight=y_wgts)

    def predict(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        x_array = self.transform(x_array)
        x_array = np.hstack((x_array, x_demo))
        return self.model.predict(x_array)

    def predict_proba(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        x_array = self.transform(x_array)
        x_array = np.hstack((x_array, x_demo))

        if hasattr(self.model, 'predict_proba'):
            y_score = self.model.predict_proba(x_array)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            y_score = expit(self.model.decision_function(x_array))
        else:
            raise ValueError(f"{type(self.model).__name__} must implement either decision_function or predict_proba")
        return y_score

    def transform(self, array: np.ndarray) -> np.ndarray:
        selector = self.__class__.selectors.get(self.fold)
        if selector is None:
            raise ValueError(f"No selector found")
        array = selector.transform(array)

        scaler = self.__class__.scalers.get(self.fold)
        if scaler is not None:
            array = scaler.transform(array)
        return array


class IterEnsemble:
    """
    IterEnsemble represents an ensemble of FoldUnit models from a single Optuna trial iteration.

    It aggregates fold-level evaluation scores, provides averaged summary statistics,
    and supports ensemble-level probability prediction via averaging.

    Attributes
    ----------
    units : List[FoldUnit]
        List of FoldUnit instances, each corresponding to a single fold in cross-validation.
    iter : int
        Index of the Optuna iteration associated with this ensemble.
    folds : List[int]
        List of fold indices used in this ensemble.
    eval_scores : List[dict]
        List of evaluation scores (Eval, Weighted) from each FoldUnit.
    summary : dict
        Dictionary containing averaged performance metrics across folds.
    optuna_score : Union[float, None]
        Score used by Optuna to evaluate this iteration (usually from the selection metric).
    hyperparameters : Union[Dict, None]
        Hyperparameter set used to initialize models in this ensemble.

    Methods
    -------
    predict_proba(x_array, x_demo)
        Predicts class probabilities by averaging outputs from all FoldUnit models.
    make_summary()
        Computes mean and standard deviation of metrics across folds for summary reporting.
    """

    def __init__(self, units: List[FoldUnit], iter_idx: int):
        self.units = units
        self.iter = iter_idx
        self.folds = [unit.fold for unit in self.units]
        self.eval_scores = [unit.scores[('Eval', 'Weighted')] for unit in self.units]
        self.summary = self.make_summary()
        self.optuna_score = None
        self.hyperparameters = None

    def __repr__(self):
        return f"<IterEnsemble(iter={self.iter}, folds={self.folds})>"

    def __str__(self):
        scores = self.summary.get('AvTotal', {})
        score_str = ', '.join(f"{k}: {v[0]:.4f}±{v[1]:.4f}" for k, v in scores.items())
        return f"IterEnsemble #{self.iter} | Eval Scores: {score_str}"

    def predict(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        predictions = [
            model.predict(x_array, x_demo) for model in self.units
        ]
        y_pred = np.column_stack(predictions).mean(axis=1)
        return y_pred

    def predict_proba(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        predictions = [
            model.predict_proba(x_array, x_demo) for model in self.units
        ]
        y_score = np.column_stack(predictions).mean(axis=1)
        return y_score

    def make_summary(self) -> Dict[str, Dict]:
        total_metrics = defaultdict(list)
        sign_metrics = defaultdict(lambda: defaultdict(list))  # {sign_key: {metric: [values]}}

        for score in self.eval_scores:
            for metric, value in score['Total'].items():
                total_metrics[metric].append(value)

            for sign_key, metrics in score['Sign'].items():
                for metric, value in metrics.items():
                    sign_metrics[sign_key][metric].append(value)

        av_total = {metric: (np.round(np.mean(vals), 5), np.round(np.std(vals), 5)) for metric, vals in total_metrics.items()}
        av_sign = {
            sign_key: {
                metric: (np.round(np.mean(vals), 5), np.round(np.std(vals), 5)) for metric, vals in metrics.items()
            }
            for sign_key, metrics in sign_metrics.items()
        }

        return {
            'AvTotal': av_total,
            'AvSign': av_sign
        }


class TuneArmy:
    """
    Stores and manages results from Optuna trials.

    This class holds a collection of IterEnsemble instances, each corresponding to
    a single hyperparameter optimization trial. It provides functionality to
    retrieve the best-performing ensemble based on the Optuna selection metric.
    Somewhat useless if you do not plan on saving all models. Boh.

    Attributes
    ----------
    ensembles : List[IterEnsemble]
        A list of IterEnsemble objects, each representing an Optuna trial.
    iter_scores : List[float]
        List of Optuna scores (selection metric values) from each ensemble.

    Methods
    -------
    best()
        Returns the IterEnsemble with the highest Optuna score.
    """

    def __init__(self, iter_ensembles: List[IterEnsemble]):
        self.ensembles = iter_ensembles
        self.iter_scores = [ens.optuna_score for ens in iter_ensembles]

    def best(self):
        """Return the ensemble with the highest optuna_score."""
        if not self.ensembles:
            return None
        return max(self.ensembles, key=lambda e: e.optuna_score)


class DatasetManager:
    """
    Handles dataset preparation, including descriptor processing, demo features, labels,
    sample weights, cross-validation splits, and signature metadata. Also performs feature
    selection and scaling per fold for consistent preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all necessary data columns.
    desc_col : str
        Name of the column containing molecular descriptors.
    demo_col : str, optional
        Column name for demographic features (default is 'DemoFP').
    label_col : str, optional
        Column name for target labels (default is 'Label').
    weight_col : str, optional
        Column name for label/sample weights (default is 'Label_weight').
    fold_col : str, optional
        Column name specifying cross-validation fold splits (default is 'Fold').
    sign_col : str, optional
        Column name for signature metadata (default is 'Signature').
    sign_names : List[str], optional
        List of names for signature dimensions (default is ['Sex', 'Age', 'Weight']).
    selector_threshold : float, optional
        Threshold for variance-based feature selection (default is 1e-3).
    scaler_method : str, optional
        Scaling method: one of {'z_score', 'min-max', 'iqr'} (default is 'iqr').

    Attributes
    ----------
    x_array : np.ndarray
        Descriptor matrix for all samples.
    x_demo : np.ndarray
        Demographic feature matrix.
    y_true : np.ndarray
        Target label array.
    y_wgts : np.ndarray
        Sample weight array.
    y_sign : IndexedDict
        Dictionary of signature metadata arrays.
    folds : List[int]
        List of unique fold identifiers.
    train_idxs : Dict[int, np.ndarray]
        Mapping from fold to training sample indices.
    eval_idxs : Dict[int, np.ndarray]
        Mapping from fold to evaluation sample indices.
    selectors : Dict[int, VarianceThreshold]
        Variance selectors fitted on training data for each fold.
    scalers : Dict[int, Scaler]
        Scalers fitted on training data per fold (only for certain descriptors).

    Methods
    -------
    get_train_eval_data(fold)
        Returns training and evaluation sets for the specified fold.
    get_full_data()
        Returns the full dataset as a dictionary.
    make_selector(x_train)
        Creates a VarianceThreshold selector and transforms the training data.
    make_scaler(x_train)
        Fits and returns a scaler based on the selected method.
    """

    def __init__(self, df: pd.DataFrame, desc_col: str, demo_col: str = 'DemoFP', label_col: str = 'Label',
                 weight_col: str = 'Label_weight', fold_col: str = 'Fold', sign_col: str = 'Signature',
                 sign_names: List[str] = None, selector_threshold: float = 1e-3, scaler_method: str = 'iqr'):

        self.desc_col = desc_col
        self.demo_col = demo_col
        self.label_col = label_col
        self.weight_col = weight_col
        self.fold_col = fold_col
        self.sign_col = sign_col
        self.sign_names = sign_names if sign_names is not None else ['Sex', 'Age', 'Weight']
        self.selector_threshold = selector_threshold
        self.scaler_method = scaler_method

        self.x_array = np.vstack(df[self.desc_col].to_numpy())
        self.x_demo = np.vstack(df[self.demo_col].to_numpy())

        self.y_true = np.vstack(df[self.label_col].to_numpy()).reshape(-1)
        self.y_wgts = np.vstack(df[self.weight_col].to_numpy()).reshape(-1)

        self.splits = np.vstack(df[self.fold_col].to_numpy()).reshape(-1)
        self.folds = sorted(df[self.fold_col].unique())

        self.y_sign = IndexedDict({key: np.array(values) for key, values in zip(sign_names, list(zip(*df[self.sign_col].tolist())))})

        self.train_idxs = {fold: np.where(self.splits != fold)[0] for fold in self.folds}
        self.eval_idxs = {fold: np.where(self.splits == fold)[0] for fold in self.folds}

        FoldUnit.train_idxs = self.train_idxs
        FoldUnit.eval_idxs = self.eval_idxs

        self.selectors = {}
        self.scalers = {}

        for fold in self.folds:
            train_idx = self.train_idxs[fold]
            x_train = self.x_array[train_idx, :]

            x_train, selector = self.make_selector(x_train)
            self.selectors[fold] = selector
            FoldUnit.selectors[fold] = selector

            if self.desc_col in ['RDKit', 'Mordred', 'CDDD', 'ChemBERTa']:
                scaler = self.make_scaler(x_train)
                self.scalers[fold] = scaler
                FoldUnit.scalers[fold] = scaler

    def get_train_eval_data(self, fold: int) -> Dict[str, Dict[str, np.ndarray]]:

        train_idx = self.train_idxs[fold]
        eval_idx = self.eval_idxs[fold]

        train_eval_data = {
            'Train': {
                'X': self.x_array[train_idx, :],
                'demo': self.x_demo[train_idx, :],
                'y': self.y_true[train_idx],
                'wgts': self.y_wgts[train_idx],
                'sign': self.y_sign[train_idx]
            },
            'Eval': {
                'X': self.x_array[eval_idx, :],
                'demo': self.x_demo[eval_idx, :],
                'y': self.y_true[eval_idx],
                'wgts': self.y_wgts[eval_idx],
                'sign': self.y_sign[eval_idx]
            }
        }

        return train_eval_data

    def get_full_data(self) -> Dict[str, np.ndarray]:

        full_data = {
            'X': self.x_array,
            'demo': self.x_demo,
            'y': self.y_true,
            'wgts': self.y_wgts,
            'sign': self.y_sign
        }
        return full_data

    def make_selector(self, x_train: np.ndarray) -> Tuple[np.ndarray, VarianceThreshold]:
        selector = VarianceThreshold(threshold=self.selector_threshold)
        x_train = selector.fit_transform(x_train)
        return x_train, selector

    def make_scaler(self, x_train: np.ndarray) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        scalers = {
            'z_score': StandardScaler(),
            'min-max': MinMaxScaler(),
            'iqr': RobustScaler(unit_variance=True)
        }
        scaler = scalers.get(self.scaler_method)
        scaler.fit(x_train)
        return scaler


def fold_evaluate(model, dataset_manager: DatasetManager, iter_idx: int) -> IterEnsemble:
    """
    Evaluate model using pre-defined folds.

    Parameters
    ----------
    model
        Instance of a model class
    dataset_manager: DatasetManager
        Instance of DatasetManager class with setup stuff :)
    iter_idx: int
        Number of iteration. For logging purposes.
    """
    units = []

    for fold in dataset_manager.folds:

        data = dataset_manager.get_train_eval_data(fold)
        unit = FoldUnit(model=deepcopy(model), fold_idx=deepcopy(fold))

        x_train, x_eval = data['Train']['X'], data['Eval']['X']
        y_train, y_eval = data['Train']['y'], data['Eval']['y']
        x_demo_train, x_demo_eval = data['Train']['demo'], data['Eval']['demo']
        y_wgts_train, y_wgts_eval = data['Train']['wgts'], data['Eval']['wgts']
        y_sign_train, y_sign_eval = data['Train']['sign'], data['Eval']['sign'] # these ARE dict[str, np.ndarray]

        unit.fit(x_train=x_train, x_demo=x_demo_train, y_array=y_train, y_wgts=y_wgts_train)

        y_pred_train = unit.predict(x_train, x_demo_train)
        y_pred_eval = unit.predict(x_eval, x_demo_eval)

        y_score_train = unit.predict_proba(x_train, x_demo_train)
        y_score_eval = unit.predict_proba(x_eval, x_demo_eval)

        unit.scores[('Train', 'Unweighted')] = outer_score(y_true=y_train, y_pred=y_pred_train, y_score=y_score_train,
                                                           y_wgts=np.ones_like(y_pred_train), y_sign=y_sign_train)
        unit.scores[('Train', 'Weighted')] = outer_score(y_true=y_train, y_pred=y_pred_train, y_score=y_score_train,
                                                         y_wgts=y_wgts_train, y_sign=y_sign_train)

        unit.scores[('Eval', 'Unweighted')] = outer_score(y_true=y_eval, y_pred=y_pred_eval, y_score=y_score_eval,
                                                          y_wgts=np.ones_like(y_pred_eval), y_sign=y_sign_eval)
        unit.scores[('Eval', 'Weighted')] = outer_score(y_true=y_eval, y_pred=y_pred_eval, y_score=y_score_eval,
                                                        y_wgts=y_wgts_eval, y_sign=y_sign_eval)

        units.append(unit)

    return IterEnsemble(units, iter_idx=iter_idx)


def ensemble_evaluate(ensemble: IterEnsemble, test_df: pd.DataFrame, desc_col: str, demo_col: str = 'DemoFP',
                      label_col: str = 'Label', weight_col: str = 'Label_weight', sign_col: str = 'Signature',
                      sign_names: List[str] = None):

    x_array = np.vstack(test_df[desc_col].to_numpy())
    x_demo = np.vstack(test_df[demo_col].to_numpy())

    y_true = np.vstack(test_df[label_col].to_numpy()).reshape(-1)
    y_wgts = np.vstack(test_df[weight_col].to_numpy()).reshape(-1)

    y_sign = IndexedDict(
        {key: np.array(values) for key, values in zip(sign_names, list(zip(*test_df[sign_col].tolist())))})

    y_score = ensemble.predict_proba(x_array, x_demo)
    y_pred = (y_score >= 0.5).astype(int)

    pred_df = pd.DataFrame({
        'SMILES': test_df['SMILES'],
        'Signature': test_df[sign_col],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score,
    })

    test_scores = {
        'Unweighted': outer_score(y_true=y_true, y_pred=y_pred, y_score=y_score,
                                  y_wgts=np.ones_like(y_pred), y_sign=y_sign),
        'Weighted': outer_score(y_true=y_true, y_pred=y_pred, y_score=y_score,
                                y_wgts=y_wgts, y_sign=y_sign)
    }
    return pred_df, test_scores


def optuna_hyperparameter_search(model_class, dataset_manager: DatasetManager, test_fold: int,
                                 selection_metric: str = 'HarmRS', n_trials: int = 32,
                                 n_jobs: int = 1, save_dir: str = None):

    print('Beginning optuna optimization search')

    if save_dir is not None:
        save_dir = save_dir.rstrip('/') + '/'

    search_params = {
        'XGBClassifier': get_params_xgb_classifier,
        'RandomForestClassifier': get_params_random_forest_classifier,
        'SVC': get_params_svc,
        'LogisticRegression': get_params_logistic_regression
    }

    if selection_metric in ['Recall', 'Accuracy', 'ROC AUC', 'Precision', 'F1 Score',
                            'MCC', 'R2', 'Balanced Accuracy', 'Specificity', 'GeomRS', 'HarmRS']:
        direction = 'maximize'
        default = -1
    else:
        direction = 'minimize'
        default = 1

    iters = []

    def objective(trial: optuna.Trial):

        model_name = model_class.__name__
        model_params_fn = search_params.get(model_name)

        if model_params_fn is None:
            raise ValueError(f'No hyperparameters defined for: {model_name}')

        try:
            model_params = model_params_fn(trial)
            model_params['n_jobs'] = n_jobs
            model = model_class(**model_params)
        except (ValueError, TypeError) as e:
            print(f"Trial {trial.number} pruned due to invalid parameters: {e}")
            raise optuna.TrialPruned() from e

        try:
            iter_ensemble = fold_evaluate(
                model=model,
                dataset_manager=dataset_manager,
                iter_idx=trial.number
            )

        except Exception as e:
            print(f"Trial {trial.number} pruned with params: {model_params} due to\n{e}")
            raise optuna.TrialPruned() from e

        trial_mean, trial_std = iter_ensemble.summary.get('AvTotal', {}).get(selection_metric, (default, 0.0))

        iter_ensemble.optuna_score = trial_mean
        iter_ensemble.hyperparameters = model_params

        iters.append(iter_ensemble)

        return trial_mean

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    tune_army = TuneArmy(iters)
    ensemble = tune_army.best()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        study_path = os.path.join(save_dir, f'study_tf_{test_fold}.pkl')
        ensemble_path = os.path.join(save_dir, f'ensemble_tf_{test_fold}.pkl')

        pickle.dump(study, open(study_path, 'wb'))
        pickle.dump(ensemble, open(ensemble_path, 'wb'))

    print('Optuna search finished successfully')

    return ensemble


parser = argparse.ArgumentParser(prog="Eval", description="Evaluate a single variant of DRID dataset")

parser.add_argument("-i", "--input", help="Path to directory with input data")
parser.add_argument("-m", "--model", help="Name of the model to use")
parser.add_argument("-T", "--type", help="Dataset type to be used.  One of: {primary, secondary}")
parser.add_argument("-M", "--metric", help="DPA metric used. One of: {prr, ror, ic}.")
parser.add_argument("-P", "--pt", help="PT set used. One of: {cred, card, cvas}.")
parser.add_argument("-f", "--test_fold", help="Index of the test fold")
parser.add_argument("-D", "--desc", help="Descriptor name to be used")
parser.add_argument("-n", "--n_trials", help="Number of optuna trials to run")
parser.add_argument("-j", "--n_jobs", help="Number of CPUs assigned to model")
parser.add_argument("-o", "--output", help="Path to the output directory")

parsed_args = parser.parse_args()


def main(args):

    sign_names = ['Sex', 'Age', 'Weight']

    input_dir = args.input
    output_dir = args.output

    model_name = args.model
    dataset_type = args.type
    dpa_metric = args.metric
    pt_set = args.pt
    test_fold = int(args.test_fold)

    desc_col = args.desc
    n_trials = int(args.n_trials)
    n_jobs = int(args.n_jobs)

    #data_path = os.path.join(input_dir, dataset_type, dpa_metric, f'drid_{pt_set}.pkl')
    data_path = os.path.join(input_dir, dataset_type, dpa_metric, f'drid_{pt_set}.joblib')
    desc_path = os.path.join(input_dir, 'descriptors.pkl')

    #data = pickle.load(open(data_path, "rb"))
    data = joblib.load(data_path)
    desc = pickle.load(open(desc_path, "rb"))

    desc = remove_inf_df(desc, desc_col)
    desc = remove_nan_df(desc, desc_col)
    desc = remove_zero_var_df(desc, desc_col)

    df = data.merge(desc[["SMILES", desc_col]], on='SMILES', how='inner')

    train_df = df[df['Fold'] != test_fold].reset_index(drop=True)
    test_df = df[df['Fold'] == test_fold].reset_index(drop=True)

    train_manager = DatasetManager(
        df=train_df,
        desc_col=desc_col,
        sign_names=sign_names
    )

    ensemble = optuna_hyperparameter_search(
        model_class=get_model(model_name),
        dataset_manager=train_manager,
        test_fold=test_fold,
        selection_metric='HarmRS',
        n_trials=n_trials,
        n_jobs=n_jobs,
        save_dir=output_dir,
    )

    pred_df, test_scores = ensemble_evaluate(
        ensemble=ensemble,
        test_df=test_df,
        desc_col=desc_col,
        sign_names=sign_names
    )

    pred_path = os.path.join(output_dir, f'preds_tf_{test_fold}.pkl')  # predictions on the "test" fold
    scores_path = os.path.join(output_dir, f'scores_tf_{test_fold}.pkl')  # scores on the "test" fold

    pickle.dump(pred_df, open(pred_path, 'wb'))
    pickle.dump(test_scores, open(scores_path, 'wb'))

if __name__ == '__main__':
    main(parsed_args)
