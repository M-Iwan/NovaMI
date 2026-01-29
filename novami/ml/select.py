import copy

import numpy as np
import pandas as pd

from novami.ml.evaluate import kf_evaluate
from novami.ml.score import average_scores


def seq_feature_selection(model, df: pd.DataFrame, features_col: str, target_col: str, n_features: int, metric: str,
                          task: str = 'classification', library: str = 'sklearn', direction: str = 'forward',
                          patience: int = 5):
    """
    Function for finding a subset of relevant samples. The found optimal subset of features will be added to initial
    dataframe as features_col + _opt. features_col + _int will be used internally.

    Parameters
    ----------
    model
        Instance of a model exposing .fit and .predict methods accepting array-like data
    df: pd.DataFrame
        A DataFrame holding the dataset
    features_col: str
        Name of a column holding feature values. Features should be numpy ndarray
    target_col: str
        Name of a column holding target values.
    n_features: int
        Number of features to find (direction = 'forward') or remove (direction = 'backward')
    metric: str
        Which metric to use as a main improvement criterion.
    task: str
        Either classification or regression
    library: str
        Which type of library model is based on. Possible option are: sklearn, torch, torch_geometric.
    direction: str
        Whether to perform a 'forward' or 'backward' feature selection
    patience: int
        Number of iterations to proceed without improvement

    Returns
    -------
    results: dict
        A dictionary holding the results of feature selection:
        * 'data' - final dataframe with selected features
        * 'scores' - iteration-wise best scores
        * 'selected_ids' - ids of selected/removed features in forward/backward directions, respectively
        * 'remaining_ids' - ids of skipped/kept features in forward/backward directions, respectively

    df: pd.DataFrame
        A DataFrame with added new column
    scores: list[float]
        A list holding changes about selected metric
    """

    if task == 'classification':
        assert metric in ['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'mcc']
    elif task == 'regression':
        assert metric in ['r2', 'mae', 'rmse']

    assert callable(getattr(model, 'fit'))
    assert callable(getattr(model, 'predict'))

    if library != 'sklearn':
        raise NotImplementedError

    if metric in ['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'mcc', 'r2']:
        objective = 'maximize'
        best_score = -1 * np.inf
    else:
        objective = 'minimize'
        best_score = np.inf

    tot_num_features = df[features_col][0].shape[0]
    num_entries = len(df)
    init_patience = patience

    df_ = df[[features_col, target_col]].copy()  # copy only features and target values

    iter_col = features_col + '_iter'
    scores = []

    feature_array = np.vstack(df_[features_col].to_numpy())

    selected_ids = []  # Changed to list to preserve order
    remaining_ids = list(range(tot_num_features))  # Changed to list to easily index into remaining features

    best_selected_ids = selected_ids.copy()
    best_remaining_ids = remaining_ids.copy()

    for iteration in range(n_features):
        print(f'  > Starting iteration: {iteration + 1} <  ')

        if patience == 0:
            print(f'  > Ending SFS due to no improvements <  ')
            break

        iter_metrics = []

        for idx in remaining_ids:
            if direction == 'forward':
                current_ids = np.array(sorted(selected_ids + [idx]), dtype='int32')
            elif direction == 'backward':
                current_ids = np.array(sorted(set(remaining_ids) - {idx}), dtype='int32')
            else:
                raise ValueError(f'Possible options for direction are: forward, backward')

            model_ = copy.deepcopy(model)
            df_.loc[:, iter_col] = np.vsplit(feature_array[:, current_ids], num_entries)
            df_.loc[:, iter_col] = df_[iter_col].apply(lambda array: array.reshape(-1))

            current_scores, _ = kf_evaluate(model_, df_, iter_col, target_col, task, library)
            av_current_scores = average_scores(current_scores)[metric]
            iter_metrics.append(av_current_scores[0])

        if objective == 'maximize':
            iter_score = np.max(iter_metrics)
            iter_idx = np.argmax(iter_metrics)
        elif objective == 'minimize':
            iter_score = np.min(iter_metrics)
            iter_idx = np.argmin(iter_metrics)
        else:
            raise ValueError(f'Unrecognized objective')

        iter_feature = remaining_ids[iter_idx]  # Get the correct feature index from remaining_ids

        if objective == 'maximize' and iter_score <= best_score:
            patience -= 1
        elif objective == 'minimize' and iter_score >= best_score:
            patience -= 1
        else:
            best_score = iter_score
            patience = init_patience
            selected_ids.append(iter_feature)  # Append to maintain order
            remaining_ids.remove(iter_feature)
            best_selected_ids = selected_ids.copy()
            best_remaining_ids = remaining_ids.copy()

        scores.append(iter_score)
        print(f'     > Current {metric}: {iter_score} <     ')

    if direction == 'forward':
        df.loc[:, features_col + '_opt'] = np.vsplit(np.vstack(df[features_col].to_numpy())[:, np.array(best_selected_ids, dtype='int32')], num_entries)
    elif direction == 'backward':
        df.loc[:, features_col + '_opt'] = np.vsplit(np.vstack(df[features_col].to_numpy())[:, np.array(best_remaining_ids, dtype='int32')], num_entries)

    df.loc[:, features_col + '_opt'] = df[features_col + '_opt'].apply(lambda array: array.reshape(-1))

    results = {'data': df, 'scores': scores, 'selected_ids': best_selected_ids, 'remaining_ids': best_remaining_ids}

    return results