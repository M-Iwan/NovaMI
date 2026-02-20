import os
import joblib
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from novami.io import write_pl
from novami.data.cluster import cc_cluster
from novami.data.transform import get_transformer_params, DataTransformer
from novami.data.partition import minimal_train_test_split, group_kfold_split
from novami.ml.evaluate import tt_evaluate, kf_evaluate
from novami.ml.utils import log


def good_curve(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str, cluster_features_col: str, training_features_col: str,
               target_col: str, evaluation: str, output_dir: str, min_distance: float, max_distance: float, step: float,
               distance_metric: str, model_name: str = 'XGBRegressor', model_params: dict= None, weights_col: str = None,
               groups_col: str = None, n_folds: int = 5, task: str = 'regression', n_jobs: int = 1):

    """
    Generate the GOOD (Generalisation Out-Of Distribution) curve.

    Parameters
    ----------
    df: pl.DataFrame
        A polars dataframe with features for clustering and training.
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
        For 'train-test' evaluation, the smallest clusters are assigned to the test set.
        For 'k-fold' evaluation, folds are prepared using GroupKFold from sklearn.
    output_dir: str
        Base path to saving directory. If None, things are not saved.
    min_distance: float
        Lower bound on the evaluated distance range.
    max_distance: float
        Upper bound on the evaluated distance range.
    step: float
        Step to take for distance evaluation.
    distance_metric: str
        Metric to be used for distance calculation based on provided features
    model_name: str
        Name of the model to use for performance evaluation
    model_params: dict[str, Any]
        Model hyperparameters to be used.
    weights_col: str
        Name of the column with sample weights for training
    groups_col: str
        Name of the column with groups to be used for separate evaluation of models
    n_folds: int
        Number of folds to prepare.
        Only relevant when evaluation = 'k-fold'
    task: str
        Either 'regression' or 'classification'
    n_jobs: int
        Number of CPUs to use during clustering and training

    Returns
    -------
    scores: pl.DataFrame
        Polars DataFrame with models metrics on the test set at a given distance threshold
    """

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars DataFrame, got {type(df)} instead")

    log("Checking columns")
    req_cols = [smiles_col, cluster_features_col, training_features_col, target_col]

    if groups_col is not None:
        req_cols.append(groups_col)
    if weights_col is not None:
        req_cols.append(weights_col)

    if missing_cols := [col for col in req_cols if col not in df.columns]:
        raise KeyError(f"Missing required columns {missing_cols}")

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise TypeError("Number of jobs must be a positive integer")

    thresholds = np.arange(min_distance, max_distance, step, dtype='float')

    data_dir = str(os.path.join(output_dir, f"{cluster_features_col}_{distance_metric}"))
    results_dir = str(os.path.join(data_dir, f"{model_name}_{training_features_col}"))
    os.makedirs(results_dir, exist_ok=True)
    scores = []

    # Transform the data prior to clustering
    dt = DataTransformer(**get_transformer_params(cluster_features_col))
    cluster_df = dt.fit_transform_df(df[[smiles_col, cluster_features_col]], features_col=cluster_features_col)

    for threshold in thresholds:
        threshold = np.round(threshold, 2)
        log(f"Processing threshold: < {threshold:.2f} >")

        log("\tClustering data")
        cluster_path = str(os.path.join(data_dir, f"cluster_{threshold:.2f}.joblib"))
        threshold_path = str(os.path.join(results_dir, f"scores_{threshold:.2f}.tsv"))

        try:
            if os.path.isfile(cluster_path):
                thr_df = df.join(joblib.load(cluster_path), how='inner', on=smiles_col)

            else:
                clustered_df = cc_cluster(
                    df=cluster_df,
                    features_col=cluster_features_col,
                    metric=distance_metric,
                    threshold=threshold,
                    n_jobs=n_jobs
                )[[smiles_col, "Cluster"]]
                joblib.dump(cluster_df, cluster_path)
                thr_df = df.join(clustered_df, on=smiles_col, how='left')

        except Exception as e:
            log(f"Unable to cluster data at threshold < {threshold:.2f} > due to:\n{e}")
            break

        if evaluation == 'train-test':

            log(f"\tSplitting data")
            try:
                thr_df = minimal_train_test_split(
                    df=thr_df,
                    fraction=0.2,
                    tolerance=step,
                    cluster_col="Cluster"
                )

            except Exception as e:
                log(f"Unable to split data into train and test set with required tolerance < {step} > at"
                    f"threshold < {threshold:.2f} > due to:\n{e}")
                break

            log(f"\tEvaluating model")
            try:
                unit = tt_evaluate(
                    model_name=model_name,
                    df=thr_df,
                    smiles_col=smiles_col,
                    features_col=training_features_col,
                    target_col=target_col,
                    set_col='Set',
                    weights_col=weights_col,
                    groups_col=groups_col,
                    n_jobs=n_jobs,
                    model_params=model_params,
                    task=task
                )

            except Exception as e:
                log(f"Unable to evaluate the model at threshold < {threshold:.2f} > due to:\n{e}")
                break

            thr_scores = unit.metrics['Testing'].with_columns(
                pl.lit(threshold).alias('Threshold')
            )

            scores.append(thr_scores)

        elif evaluation == 'k-fold':

            log(f"\tSplitting data")
            try:
                thr_df = group_kfold_split(
                    df=thr_df,
                    cluster_col='Cluster',
                    n_folds=n_folds
                )

            except Exception as e:
                log(f"Unable to split data into < {n_folds} > folds with required tolerance < {step} > at"
                    f"threshold < {threshold:.2f} > due to:\n{e}")
                break

            log(f"\tEvaluating model")
            try:
                ensemble = kf_evaluate(
                    model_name=model_name,
                    df=thr_df,
                    smiles_col=smiles_col,
                    features_col=training_features_col,
                    target_col=target_col,
                    fold_col="Fold",
                    weights_col=weights_col,
                    groups_col=groups_col,
                    n_jobs=n_jobs,
                    model_params=model_params,
                    task=task
                )

            except Exception as e:
                log(f"Unable to evaluate the model at threshold < {threshold:.2f} > due to:\n{e}")
                break

            thr_scores = ensemble.metrics["Testing"].with_columns(
                pl.lit(threshold).alias('Threshold')
            )

            scores.append(thr_scores)
            log(f"\tEvaluation successful.")

        else:
            raise ValueError(f"Unknown evaluation {evaluation}")

        write_pl(thr_scores, threshold_path)
        log(f"\tResults written to\n {threshold_path}")

    if scores:
        scores = pl.concat(scores)

        write_pl(scores, str(os.path.join(results_dir, "scores.tsv")))
        return scores

    else:
        print(f"Unable to obtain even one valid split.")
        return None
