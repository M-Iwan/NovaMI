import os, sys
import joblib
from typing import Union, List

import pandas as pd
import polars as pl

from novami.data.partition import cc_kfold_split
from novami.data.manager import DatasetManager
from novami.ml.params import get_params_function
from novami.ml.optimize import optimize_unit_optuna


def good_curve(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str, cluster_features_col: str, training_features_col: str,
               target_col: str, evaluation: str, base_dir: str, groups_col: str = None, weights_col: str = None,
               strat_col: str = None, distance_threshold: float = 0.3, distance_metric: str = 'jaccard', n_folds: int = 5,
               tolerance: float = 0.2, model_name: str = 'XGBRegressor', optim_metric: str = 'RMSE',
               task: str = 'regression', n_trials: int = 64, n_jobs: int = 1):

    """
    Generate the GOOD (Generalisation Out-Of Distribution) curve.

    Parameters
    ----------
    df: Union[pd.DataFrame, pl.DataFrame]
        Either pandas or polars dataframe with features for clustering and training
    smiles_col: str
        Name of the column with SMILES strings
    cluster_features_col: str
        Name of the column with features to be used for clustering
    training_features_col: str
        Name of the column with features to be used for training
    target_col: str
        Name of the column with target values
    evaluation: str
        How to evaluate models: 'train-test', 'k-fold'
        In both cases, data are first clustered using Connected Components algorithm.
        For 'train-test' evaluation the smallest clusters are assigned to the test set.
        For 'k-fold' evaluation n_folds are prepared using StratifiedGroupKFold class from sklearn.
    base_dir: str
        Base path to saving directory
    groups_col: str
        Name of the column with groups to be used for separate evaluation of models
    weights_col: str
        Name of the column with sample weights for training
    strat_col: str
        Name of the column to be used for stratification during splitting.
        Only relevant when evaluation = 'k-fold'
    distance_threshold: float
        Minimum distance between training and test sets
    distance_metric: str
        Metric to be used for distance calculation based on provided features
    n_folds: int
        Number of folds to prepare.
        Only relevant when evaluation = 'k-fold'
    tolerance: float
        Maximum acceptable Relative Standard Deviation between fold sizes.
        Only relevant when evaluation = 'k-fold'
    model_name: str
        Name of the model to use for performance evaluation
    optim_metric: str
        Name of the metric to use for optimization guidance
    task: str
        Type of model. 'regression' or 'classification'
    n_trials: int
        Number of Optuna optimization trials to run
    n_jobs: int
        Number of CPUs to use during clustering and training

    Returns
    -------
    scores: pl.DataFrame
        Polars DataFrame with models metrics on the test set at a given distance threshold
    """

    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(f"Expected either pandas or polars DataFrame, got {type(df)} instead")

    if is_pandas := isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    req_cols = [smiles_col, cluster_features_col, training_features_col, target_col]

    if groups_col is not None:
        req_cols.append(groups_col)
    if weights_col is not None:
        req_cols.append(weights_col)
    if strat_col is not None:
        req_cols.append(strat_col)

    if missing_cols := [col for col in req_cols if col not in df.columns]:
        raise KeyError(f"Missing required columns {missing_cols}")

    prm_fn = get_params_function(model_name)
    if prm_fn is None:
        raise ValueError(f"Model < {model_name} > not supported. Please add a corresponding parameter function to novami.ml.params")

    if optim_metric not in ["TP", "FP", "FN", "TN", "Accuracy", "Recall", "Specificity", "Precision",
                          "Balanced Accuracy", "GeomRS", "HarmRS", "F1 Score", "ROC AUC", "MCC"]:
        raise ValueError(f"Metric < {optim_metric} > not supported")

    if not isinstance(n_trials, int) or n_trials < 1:
        raise TypeError("Number of trials must be a positive integer")

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise TypeError("Number of jobs must be a positive integer")

    if evaluation == 'k-fold':
        try:
            df = cc_kfold_split(df=df, strat_col=strat_col, smiles_col=smiles_col, features_col=cluster_features_col,
                                threshold=distance_threshold, metric=distance_metric, n_folds=n_folds, n_jobs=n_jobs,
                                tolerance=tolerance)

        # Unable to split data within given tolerance
        except RuntimeError as e:
            print(e)
            return None

        except Exception as e:
            print(f"Unexpected error encountered while partitioning data:\n{e}")
            return None

    elif evaluation == 'train-test':
        try:


    save_dir = os.path.join(base_dir, cluster_features_col, distance_metric, f"{distance_threshold}:.2f")
    os.makedirs(save_dir, exist_ok=True)

    df_path = os.path.join(save_dir, f"folds.joblib")
    joblib.dump(df[[smiles_col, target_col, "Cluster", "Fold"]], df_path)

    all_scores = []

    # ML evaluation loop
    for fold in df["Fold"].unique():

        dataset_manager = DatasetManager(
            df=df,
            smiles_col=smiles_col,
            features_col=features_col,
            target_col=target_col,
            fold_col="Fold",
            test_fold=fold,
            weights_col=weights_col,
            groups_col=groups_col,
        )

        study, ensemble = optimize_unit_optuna(
            model_name=model_name,
            dataset_manager=dataset_manager,
            optimization_metric=optim_metric,
            task=task,
            n_trials=n_trials,
            n_jobs=n_jobs
        )

        res_dir = os.path.join(save_dir, f"results")
        os.makedirs(res_dir, exist_ok=True)

        study_path = os.path.join(res_dir, f"study_tf_{fold}.joblib")
        ensemble_path = os.path.join(res_dir, f"ensemble_tf_{fold}.joblib")

        joblib.dump(study, study_path)
        joblib.dump(ensemble, ensemble_path)

        test_data, test_smiles = dataset_manager.get_test_data(), dataset_manager.get_test_smiles()
        y_true = test_data['y_true']
        y_pred = test_data['x_array']

        pred = pl.DataFrame({smiles_col: test_smiles, 'y_true': y_true, 'y_pred': y_pred})
        scores = ensemble.score(**test_data)

        pred_path = os.path.join(res_dir, f"pred_tf_{fold}.joblib")
        scores_path = os.path.join(res_dir, f"scores_tf_{fold}.joblib")

        joblib.dump(pred, pred_path)
        joblib.dump(scores, scores_path)

        all_scores.append(scores)