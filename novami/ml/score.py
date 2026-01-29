"""
Somewhat deprecated module. The scoring are now shipped together with Classifier and Regressor Units.
"""

from collections import defaultdict
from typing import Optional, Dict

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error


def score_classification_model(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                               sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Used with 1D numpy arrays.

    Parameters
    ----------

    y_true: np.ndarray
        Array of true labels
    y_pred: np.ndarray
        Array of predicted labels.
    y_score: np.ndarray
        Array of predicted probabilities (from .predict_proba or .decision_function)
    sample_weight: np.ndarray
        Array of weights. Optional.
    """

    def safe_div(numerator, denominator, default_=0.0):
        return numerator / denominator if denominator != 0 else default_

    if sample_weight is None:  # we don't know the weights
        sample_weight = np.ones_like(y_true)  # just say they are equal and don't think about it anymore!

    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

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
        'ROC AUC': np.round(roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=sample_weight), 5),
        'MCC': np.round(safe_div((tp * tn) - (fp * fn), np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), 5)
    }

    return metrics

def score_regression_model(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None):

    if sample_weight is not None:
        scores = {
            'R2': r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
            'MAE': mean_absolute_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
            'RMSE': mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False, sample_weight=sample_weight)
        }
    else:
        scores = {
            'R2': r2_score(y_true=y_true, y_pred=y_pred),
            'MAE': mean_absolute_error(y_true=y_true, y_pred=y_pred),
            'RMSE': mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
        }

    return scores


def average_scores(scores: dict) -> dict:
    """
    Calculate the average metric values separately for normal and weighted cross-validated models.

    Parameters
    ----------
    scores : dict
        An output from kf_evaluate, bootstrap_evaluate, or test_evaluate.
        Keys should be in the format 'model_{i}' for unweighted or 'model_{i}_w' for weighted.

    Returns
    -------
    av_scores : dict[str, dict[str, List[float]]]
        A dictionary with 'unweighted' and 'weighted' keys,
        each mapping to a dict of metric names and their [mean, std].
    """

    # Separate into unweighted and weighted
    grouped_scores = {'unweighted': defaultdict(list), 'weighted': defaultdict(list)}

    for model_name, model_scores in scores.items():
        group = 'weighted' if model_name.endswith('_w') else 'unweighted'
        for metric, value in model_scores.items():
            grouped_scores[group][metric].append(value)

    # Compute means and stds
    av_scores = {
        group: {
            metric: [np.round(np.mean(values), 5), np.round(np.std(values), 5)]
            for metric, values in metrics.items()
        }
        for group, metrics in grouped_scores.items()
    }

    return av_scores