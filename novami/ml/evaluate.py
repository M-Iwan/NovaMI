import inspect
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from novami.ml.score import score_regression_model, score_classification_model

def kf_evaluate(model, df: pd.DataFrame, features_col: str, target_col: str, task: str = 'classification',
                library: str = 'sklearn', kf_params: dict = None):
    """
    model : initialized instance
    """

    sc_fs = {
        'classification': score_classification_model,
        'regression': score_regression_model
        }

    score_function = sc_fs.get(task)

    models = {}
    scores = {}

    def_params = {'n_splits': 5, 'shuffle': True, 'random_state': np.random.RandomState(0)}
    if kf_params is not None:
        def_params.update(kf_params)

    skf = StratifiedKFold(n_splits=def_params['n_splits'], shuffle=def_params['shuffle'],
                          random_state=def_params['random_state'])

    if library == 'sklearn':
        x = np.vstack(df[features_col].to_numpy())
        y = df[target_col].to_numpy()

    elif library == 'torch':
        raise NotImplementedError

    elif library == 'torch_geometric':
        x = df[features_col].values
        y = df[target_col].values
    else:
        raise ValueError('Wrong library type')

    for i, (train_index, eval_index) in enumerate(skf.split(x, y)):

        model_ = copy.deepcopy(model)
        model_.fit(x[train_index], y[train_index])

        y_pred = model_.predict(x[eval_index])
        score = score_function(y_true=y[eval_index], y_pred=y_pred)

        models[f'model_{i}'] = model_
        scores[f'model_{i}'] = copy.deepcopy(score)

    return scores, models


def fold_evaluate(model, df: pd.DataFrame, features_col: str, target_col: str, fold_col: str, weights_col: str = None,
                  task: str = 'classification'):
    """
    Evaluate model using pre-defined splits.
    Model must implement .fit and .predict methods and work on numpy ndarrays.
    """

    sc_fs = {
        'classification': score_classification_model,
        'regression': score_regression_model
        }

    score_function = sc_fs.get(task)

    x_values = np.vstack(df[features_col].to_numpy())
    y_values = np.vstack(df[target_col].to_numpy())
    splits = np.vstack(df[fold_col].to_numpy())
    folds = df[fold_col].nunique()

    scores = {}
    models = {}

    for i in range(folds):

        model_ = copy.deepcopy(model)

        train_idx = np.where(splits != i)[0]
        eval_idx = np.where(splits == i)[0]

        y_train = y_values[train_idx, :]
        y_eval = y_values[eval_idx, :]

        if y_train.shape[1] == 1:  # i.e. only one target:
            y_train = y_train.reshape(-1)
            y_eval = y_eval.reshape(-1)

        if (weights_col is not None) and ('sample_weight' in inspect.signature(model_.fit).parameters):  # i.e. does model accept weights
            weights_ = np.vstack(df[weights_col].to_numpy()).reshape(-1)[train_idx]
            model_.fit(x_values[train_idx, :], y_train, sample_weight=weights_)
            y_pred = model_.predict(x_values[eval_idx, :])
            scores[f'model_{i}_w'] = score_function(y_true=y_eval, y_pred=y_pred, sample_weight=weights_)  # log weighted scores
            scores[f'model_{i}'] = score_function(y_true=y_eval, y_pred=y_pred)  # log un-weighted scores

        else:
            model_.fit(x_values[train_idx, :], y_train)
            y_pred = model_.predict(x_values[eval_idx, :])
            scores[f'model_{i}'] = score_function(y_true=y_eval, y_pred=y_pred)

        models[f'model_{i}'] = model_

    return scores, models


def bootstrap_evaluate(model, df: pd.DataFrame, features_col: str, target_col: str, task: str = 'classification',
                       library: str = 'sklearn', bs_params: dict = None):
    """
    model : initialized instance of model following sklearn API
    Does not support sample_weights
    """

    models = dict()
    scores = dict()

    def_params = {'frac': 0.8, 'replace': True, 'random_state': np.random.RandomState(0), 'n_iter': 5}
    if bs_params is not None:
        def_params.update(bs_params)

    for i in range(def_params['n_iter']):

        df_train = df.sample(frac=def_params['frac'], replace=def_params['replace'], random_state=def_params['random_state'])
        df_eval = df[~df.index.isin(df_train.index)]

        y_train = df_train[target_col].to_numpy()
        y_eval = df_eval[target_col].to_numpy()

        if library == 'sklearn':
            x_train = np.vstack(df_train[features_col].to_numpy())
            x_eval = np.vstack(df_eval[features_col].to_numpy())

        elif library == 'torch':
            raise NotImplementedError

        elif library == 'torch_geometric':
            x_train = df_train[features_col].values  # should take the 'SMILES' column, as models have built-in vectorizer
            x_eval = df_eval[features_col].values

        else:
            raise ValueError('Wrong library')

        model_ = copy.deepcopy(model)
        model_.fit(x_train, y_train)

        if task == 'regression':
            score = score_regression_model(model_, x_eval, y_eval)
        elif task == 'classification':
            score = score_classification_model(model_, x_eval, y_eval)
        else:
            raise ValueError("Allowed options for < task > are: regression, classification")

        models[f'model_{i+1}'] = model_
        scores[f'model_{i+1}'] = copy.deepcopy(score)

    return scores, models


def test_evaluate(model, df: pd.DataFrame, features_col: str, target_col: str, task: str = 'classification',
                  library: str = 'sklearn', split_column: str = 'Dataset'):
    """
    The passed df is expected to have 'train'/'test' values in split_column argument
    Does not support sample_weights.
    """

    df_train = df[df[split_column] == 'train']
    df_test = df[df[split_column] == 'test']

    y_train = df_train[target_col].to_numpy()
    y_test = df_test[target_col].to_numpy()

    if library == 'sklearn':
        x_train = np.vstack(df_train[features_col].to_numpy())
        x_test = np.vstack(df_test[features_col].to_numpy())

    elif library == 'torch':
        raise NotImplementedError

    elif library == 'torch_geometric':
        x_train = df_train[features_col].values
        x_test = df_test[features_col].values

    else:
        raise ValueError('Wrong library passed')

    model_ = copy.deepcopy(model)
    model_.fit(x_train, y_train)

    if task == 'regression':
        score = score_regression_model(model_, x_test, y_test)
    elif task == 'classification':
        score = score_classification_model(model_, x_test, y_test)
    else:
        raise ValueError("Allowed options for < task > are: regression, classification")

    models = {'model_1': model_}
    scores = {'model_1': score}

    return scores, models