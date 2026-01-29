import inspect
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import optuna

from novami.ml.evaluate import kf_evaluate, bootstrap_evaluate, test_evaluate
from novami.ml.models import Unit, Ensemble, ClassifierUnit, RegressorUnit, ClassifierEnsemble, RegressorEnsemble
from novami.ml.score import average_scores, score_regression_model, score_classification_model
from novami.ml.params import *
from novami.ml.utils import *
from novami.data.manager import DatasetManager


def random_parameters(params: dict):
    selected = dict()
    for param, values in params.items():
        selected[param] = values[np.random.randint(0, len(values))]
    return selected


def grid_parameters(params: dict):
    combinations = list(product(*params.values()))
    grid = [{key: value for key, value in zip(params.keys(), combination)} for combination in combinations]

    return grid


def random_hyperparameter_search(model_class, df: pd.DataFrame, features_col: str, target_col: str, parameters: dict,
                                 n_iter: int = 100, task: str = 'classification', library: str = 'sklearn',
                                 evaluation: str = 'kf'):
    """
    If evaluation = 'ts' is used, the dataframe must hold a 'Dataset' column with 'train'/'test'
    literal strings that are used to split it into training/evaluation sets

    Notes
    -----
    The function is now somewhat deprecated and not integrated with the rest of the code.
    """

    results = []

    for iteration in range(n_iter):
        current_params = random_parameters(parameters[model_class.__name__])

        try:
            model = model_class(**current_params)

            if evaluation == 'kf':
                scores, _ = kf_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'bs':
                scores, _ = bootstrap_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'ts':

                assert 'Dataset' in df.columns
                dataset_values = df['Dataset'].unique()
                assert all(['train' in dataset_values, 'test' in dataset_values, len(dataset_values) == 2])

                scores, _ = test_evaluate(model, df, features_col, target_col, task, library, split_column='Dataset')
            else:
                raise ValueError(f'Expected < evaluation > to be either kf, bs, or ts, got < {evaluation} > instead)')
            av_scores = average_scores(scores)

            results.append({'parameters': current_params, 'scores': av_scores})

        except ValueError:  # skip illegal combinations of parameters
            pass

    return results


def grid_hyperparameter_search(model_class, df: pd.DataFrame, features_col: str, target_col: str, parameters: dict,
                               task: str = 'classification', library: str = 'sklearn', evaluation: str = 'kf'):
    """
    Notes
    -----
    The function is now somewhat deprecated and not integrated with the rest of the code.
    """
    results = []

    grid = grid_parameters(parameters[model_class.__name__])
    total_number = len(grid)

    print(f'> Total number of models to be evaluated: {total_number} <')

    for current_param in grid:
        try:
            model = model_class(**current_param)

            if evaluation == 'kf':
                scores, _ = kf_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'bs':
                scores, _ = bootstrap_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'ts':

                assert 'Dataset' in df.columns
                dataset_values = df['Dataset'].unique()
                assert all(['train' in dataset_values, 'test' in dataset_values, len(dataset_values) == 2])

                scores, _ = test_evaluate(model, df, features_col, target_col, task, library, split_column='Dataset')
            else:
                raise ValueError(f'Expected < evaluation > to be either kf or bs, got < {evaluation} > instead)')

            av_scores = average_scores(scores)

            results.append({'parameters': current_param, 'scores': av_scores})

        except ValueError:  # skip illegal combinations of parameters
            pass

    return results


def optuna_hyperparameter_search(model_class, df: pd.DataFrame, features_col: str, target_col: str, fold_col: str,
                                 parameters: dict = None, sample_weight_col: str = None, selection_metric: str = 'mcc',
                                 n_trials: int = 64):
    """
    Perform hyperparameter search using optuna.

    Parameters
    ----------
    model_class
        A model implementing .fit and .predict functions based on np.ndarrays
    df: pd.DataFrame
        A dataframe with features, target, and fold indices.
    features_col: str
        Name of the column with features represented as NumPy arrays.
    target_col: str
        Name of the column with target values.
    fold_col: str
        Name of the column with fold mapping for test set.
    sample_weight_col: str
        Name of the column with sample weights.
    parameters: dict[func]
        A dictionary mapping model name to functions for creating trial parameters.
    selection_metric: str
        Name of the metric to use for model selection. Default is 'mcc'.
    n_trials: int
        Number of trials to perform. Default is 64.

    Returns
    -------
    study
        The whole performed study
    best_trial
        Trial with the best final metrics.
    """

    # select task and scoring function based on passed metric
    if selection_metric in ['Recall', 'Accuracy', 'ROC AUC', 'Precision', 'F1 Score',
                            'MCC', 'Balanced Accuracy', 'Specificity']:
        score_function = score_classification_model
    elif selection_metric in ['R2', 'MAE', 'RMSE']:
        score_function = score_regression_model
    else:
        raise ValueError('Unknown selection metric passed')

    # Select optimization direction and variance penalty sign
    if selection_metric in ['Recall', 'Accuracy', 'ROC AUC', 'Precision', 'F1 Score',
                            'MCC', 'R2', 'Balanced Accuracy', 'Specificity']:
        direction = 'maximize'
    else:
        direction = 'minimize'

    x_values = np.vstack(df[features_col].to_numpy())
    y_values = np.array(df[target_col].tolist())
    splits = np.array(df[fold_col].tolist())
    n_split = df[fold_col].unique()

    def objective(trial: optuna.Trial):

        model_name = model_class.__name__
        model_params = parameters.get(model_name)

        if model_params is None:
            raise ValueError(f'Hyperparameters missing for {model_name}')

        # check for illegal combinations of parameters
        try:
            model_hps = model_params(trial)
            model = model_class(**model_hps)
        except ValueError as e:
            raise optuna.TrialPruned() from e

        scores = []
        metrics = {}  # compatibility with other code

        for fold in n_split:
            train_idx = np.where(splits != fold)[0]
            eval_idx = np.where(splits == fold)[0]

            if (sample_weight_col is not None) and ('sample_weight' in inspect.signature(model.fit).parameters):
                weights_ = np.vstack(df[sample_weight_col].to_numpy()).reshape(-1)[train_idx]
                model.fit(x_values[train_idx, :], y_values[train_idx], sample_weight=weights_)

            else:
                model.fit(x_values[train_idx, :], y_values[train_idx])

            y_pred = model.predict(x_values[eval_idx, :])

            fold_score = score_function(y_true=y_values[eval_idx], y_pred=y_pred)  # that's a dictionary
            scores.append(fold_score[selection_metric])
            metrics[f'model_{fold}'] = fold_score

            # add early stopping
            trial.report(fold_score[selection_metric], fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = np.mean(scores)

        trial.set_user_attr('metrics', average_scores(metrics))

        return mean_score

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    best_trial = study.best_trial

    return study, best_trial


def train_ensemble(unit: Unit, dataset_manager: DatasetManager, task: str = 'regression'):
    """
    Train an ensemble model based on dataset with KFold split.


    Parameters
    ----------
    unit: Unit
        A Unit to use for training.
    dataset_manager: DatasetManager
        Dataset manager
    task: str
        Regression or classification

    Returns
    -------
    ensemble: Ensemble
    """

    ensemble = {'regression': RegressorEnsemble, 'classification': ClassifierEnsemble}.get(task)()

    for fold in dataset_manager.folds:
        train_data = dataset_manager.get_train_data(fold)
        eval_data = dataset_manager.get_eval_data(fold)

        funit = deepcopy(unit)
        funit.fit(**train_data)
        funit.metrics['Training'] = funit.score(**train_data)
        funit.metrics['Validation'] = funit.score(**eval_data)
        ensemble.add_unit(funit)

    ensemble.average_metrics()
    return ensemble


def optimize_unit_optuna(model_name: str, dataset_manager: DatasetManager, optimization_metric: str = 'RMSE',
                         task: str = 'regression', n_trials: int = 64, n_jobs: int = 1):

    # Select optimization direction
    if optimization_metric in ['Recall', 'Accuracy', 'ROC AUC', 'Precision', 'F1 Score',
                               'MCC', 'R2', 'Balanced Accuracy', 'Specificity']:
        direction = 'maximize'
    else:
        direction = 'minimize'

    features = dataset_manager.features_col

    best_ensemble = None
    best_metric = float('-inf') if direction == 'maximize' else float('inf')

    def objective(trial: optuna.trial.Trial):
        nonlocal best_ensemble, best_metric

        model_params_fn = get_params_function(model_name)

        try:
            model_params = model_params_fn(trial)

            unit = prepare_unit(
                model_name=model_name,
                model_params=model_params,
                features=features,
                task=task,
                n_jobs=n_jobs,
            )

        except (ValueError, TypeError) as e:
            print(f"Trial {trial.number} pruned due to invalid parameters: {e}")
            raise optuna.TrialPruned() from e

        try:
            ensemble = train_ensemble(unit, dataset_manager, task=task)
            metric = (ensemble.metrics
                .filter(pl.col('Set') == 'Validation', pl.col('Metric') == optimization_metric, pl.col('Group') == 'Overall')
                .select('Mean').item()
            )

            is_better = (direction == 'maximize' and metric > best_metric) or \
                        (direction == 'minimize' and metric < best_metric)

            if is_better:
                best_metric = metric
                best_ensemble = ensemble
            else:
                del ensemble

        except Exception as e:
            print(f"Trial {trial.number} pruned with params: {model_params}\n{e}")
            raise optuna.TrialPruned() from e

        return metric

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    return study, best_ensemble
