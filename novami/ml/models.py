import inspect
import joblib
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import polars as pl
from scipy import stats
from scipy.special import expit
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score, mean_absolute_error, root_mean_squared_error

from novami.data.transform import DataTransformer


class Unit(ABC):
    """
    A wrapper for classical machine learning models that provides an end-to-end
    pipeline for feature selection, data preprocessing, model training, and inference.

    This abstract base class provides a framework for building standardized machine learning
    workflows with automatic data validation, feature selection, and scaling capabilities.
    Each transformation stage can be individually enabled or disabled.

    Parameters
    ----------
    model: object
        A scikit-learn compatible model with fit and predict methods.
    transformer: DataTransformer
        Transformer responsible for pre-processing training data.

    Methods
    -------
    fit(x_array, y_array, sample_weight=None, groups=None)
        Fits the preprocessing pipeline and the underlying model on training data.

    predict(x_array)
        Abstract method to be implemented by subclasses for making predictions.

    score(x_array, y_array, sample_weight=None, groups=None)
        Abstract method to be implemented by subclasses for model evaluation.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com

    Changes
    -------
    Added in version 0.1.2
    Imputation added in version 0.1.3
    Correlation analysis added in version 0.1.4
    Group scoring added in version 0.2.6
    """

    def __init__(self, model, transformer: DataTransformer):
        self.model = model
        self.name = self.model.__class__.__name__
        self.transformer = transformer

        self.metrics = {}
        self._model_fit = False
        self._has_weights = 'sample_weight' in inspect.signature(self.model.fit).parameters
        self._task = None

    def __repr__(self):
        return f"<{self.__class__.__name__} using {self.name}>"

    def __str__(self):
        return f"<{self.__class__.__name__} using {self.name}>"


    def fit(self, x_array: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            groups: Optional[any] = None):
        """
        Transform data and fit model. Groups parameter is ignored during fitting and exists for compatibility.
        """

        x_array = self.transformer.validate_features(x_array)
        y_true = self.transformer.validate_targets(y_true)

        if sample_weight is not None:
            sample_weight = self.transformer.validate_targets(sample_weight)

        x_array = self.transformer.fit_transform(x_array)

        if self._has_weights and sample_weight is not None:
            self.model.fit(x_array, y_true, sample_weight=sample_weight)
        else:
            self.model.fit(x_array, y_true)

        self._model_fit = True

    def predict(self, x_array: np.ndarray):
        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')

        x_array = self.transformer.validate_features(x_array)
        x_array = self.transformer.transform(x_array)

        predictions = self.model.predict(x_array).reshape(-1) # no support for multitask predictions

        return predictions

    @abstractmethod
    def score(self, x_array: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        pass

    def save(self, path: str):
        joblib.dump(self, path, compress=('xz', 5))

    @staticmethod
    def load(path: str):
        return joblib.load(path)


class ClassifierUnit(Unit):
    """
    Unit class for classification that additionally implements predict_proba.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Added in NovaMI version: 0.1.2
    """
    def __init__(self, model, transformer: DataTransformer, use_proba: bool = True):

        super().__init__(model=model, transformer=transformer)

        self.use_proba = use_proba
        self._has_proba = hasattr(self.model, 'predict_proba')
        self._has_decision = hasattr(self.model, 'decision_function')
        self._task = 'Classification'

    def predict_proba(self, x_array: np.ndarray):
        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')
        if not self.use_proba:
            raise RuntimeError('Probability prediction is disabled. It can be overridden by setting .use_proba = True')

        x_array = self.transformer.validate_features(x_array)
        x_array = self.transformer.transform(x_array)

        if self._has_proba:
            y_score = self.model.predict_proba(x_array)[:, 1]
        elif self._has_decision:
            y_score = expit(self.model.decision_function(x_array))
        else:
            raise RuntimeError(f'{self.name} supports neither decision_function nor predict_proba, '
                               f'but .use_proba was set to True')

        return y_score.reshape(-1)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
              sample_weight: Optional[np.ndarray] = None):

        def safe_div(numerator, denominator, default_=0.0):
            return numerator / denominator if denominator != 0 else default_

        y_true = self.transformer.validate_targets(y_true)
        y_pred = self.transformer.validate_targets(y_pred)

        if sample_weight is not None:
            sample_weight = self.transformer.validate_targets(sample_weight)
        else:
            sample_weight = np.ones_like(y_true)

        if y_score is not None:
            y_score = self.transformer.validate_targets(y_score)
            roc_auc = np.round(roc_auc_score(y_true, y_score, sample_weight=sample_weight), 5)
        else:
            roc_auc = np.nan

        conf_mat = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)

        if len(np.unique(y_true)) != 2:
            return pl.DataFrame()

        TN, FP, FN, TP = conf_mat.ravel()

        rec = safe_div(TP, TP + FN)
        spec = safe_div(TN, TN + FP)

        values = {
            'TP': np.round(TP, 5),
            'FP': np.round(FP, 5),
            'FN': np.round(FN, 5),
            'TN': np.round(TN, 5),
            'Accuracy': np.round(safe_div(TP + TN, TP + TN + FP + FN), 5),
            'Recall': np.round(rec, 5),
            'Specificity': np.round(spec, 5),
            'Precision': np.round(safe_div(TP, TP + FP), 5),
            'Balanced Accuracy': np.round((rec + spec) / 2, 5),
            'GeomRS': np.round(np.sqrt(rec * spec), 5),
            'HarmRS': np.round(safe_div(2 * rec * spec, rec + spec), 5),
            'F1 Score': np.round(safe_div(2 * TP, 2 * TP + FP + FN), 5),
            'ROC AUC': roc_auc,
            'MCC': np.round(safe_div((TP * TN) - (FP * FN), np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))), 5)
        }

        df = pl.DataFrame(values).unpivot(variable_name='Metric', value_name='Value')
        return df

    def score(self, x_array: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):

        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')

        x_array = self.transformer.validate_features(x_array)
        y_true = self.transformer.validate_targets(y_true)

        y_pred = self.predict(x_array)

        if self.use_proba and (self._has_proba or self._has_decision):
            y_score = self.predict_proba(x_array)
        else:
            y_score = None

        metrics = (self.evaluate(y_true=y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = self.transformer.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:  # If for some reason a group is empty...
                    g_y_true = y_true[mask]
                    g_y_pred = y_pred[mask]
                    g_y_score = y_score[mask] if y_score is not None else None
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = self.evaluate(y_true=g_y_true, y_pred=g_y_pred, y_score=g_y_score, sample_weight=g_sample_weight)
                    g_metrics = g_metrics.with_columns(pl.lit(group_id).alias('Group'))
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        return metrics


class RegressorUnit(Unit):
    """
    Unit class for regression.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Added in NovaMI version: 0.1.2
    """
    def __init__(self, model, transformer: DataTransformer):

        super().__init__(model=model, transformer=transformer)
        self._task = 'Regression'

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None):

        y_true = self.transformer.validate_targets(y_true)
        y_pred = self.transformer.validate_targets(y_pred)

        if sample_weight is not None:
            sample_weight = self.transformer.validate_targets(sample_weight)
        else:
            sample_weight = np.ones_like(y_true)

        values = {
            'R2': r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
            'MAE': mean_absolute_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight),
            'RMSE': root_mean_squared_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
        }

        df = pl.DataFrame(values).unpivot(variable_name='Metric', value_name='Value')
        return df

    def score(self, x_array: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')

        x_array = self.transformer.validate_features(x_array)
        y_true = self.transformer.validate_targets(y_true)

        y_pred = self.predict(x_array)

        metrics = (self.evaluate(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = self.transformer.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:  # If for some reason a group is empty...
                    g_y_true = y_true[mask]
                    g_y_pred = y_pred[mask]
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = self.evaluate(y_true=g_y_true, y_pred=g_y_pred, sample_weight=g_sample_weight)
                    g_metrics = g_metrics.with_columns(pl.lit(group_id).alias('Group'))
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        return metrics


class Ensemble(ABC):
    """
    Base class for managing ensembles of Unit models.

    Parameters
    ----------
    units : List[Unit], optional
        Pre-trained Unit instances to include in the ensemble.
    voting : str, default='soft'
        Strategy for combining predictions
    weights : np.ndarray, optional
        Weights for each unit when using weighted voting. Must be a 1D array or
        a column vector with length matching the number of units.

    Attributes
    ----------
    units: List[Unit]
        The collection of Unit instances in the ensemble.
    voting: str
        The voting strategy used for combining predictions.
    weights: np.ndarray or None
        Normalized weights for each unit (sums to 1.0).
    unit_type: type
        The type of the units in the ensemble (all units must be of the same type).
    metrics: dict
        Dictionary storing evaluation metrics for the ensemble.
    _has_units: bool
        Whether the ensemble contains any units.
    _has_weights: bool
        Whether weights have been provided for weighted voting.

    Methods
    -------
    add_unit(unit)
        Add a Unit to the ensemble.

    average_metrics()
        Calculate average metrics across all units in the ensemble.

    predict(x_array)
        Make predictions using the ensemble. Must be implemented by subclasses.

    score(x_array, y_true, sample_weight=None, groups=None)
        Evaluate the ensemble on test data. Must be implemented by subclasses.

    save(path)
        Save the ensemble to a file.

    load(path)
        Load an ensemble from a file.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Added in NovaMI version: 0.1.2
    """

    def __init__(self, units: Optional[List[Unit]] = None, voting: str = 'soft', weights: Optional[np.ndarray] = None):
        self.units = [] if units is None else units
        self.voting = voting
        self.unit_type = None
        self.metrics = {}
        self._has_units = len(self.units) > 0
        self._has_weights = False
        self._task = None

        if weights is not None:
            if not isinstance(weights, np.ndarray):
                raise TypeError('Weights must be a numpy array')

            if weights.ndim > 2:
                raise ValueError('Weights must be flattenable to a 1D array')
            if weights.ndim == 2:
                if weights.shape[1] == 1:
                    weights = weights.reshape(-1)
                else:
                    raise ValueError('2D array was passed, but the 2nd dimension is greater than 1.')
            if weights.shape[0] != len(self.units):
                raise ValueError('Number of units does not match the number of passed weights.')

            if np.sum(np.abs(weights)) == 0:
                raise ValueError('Weights cannot all be zeroes.')

            self.weights = weights / np.sum(weights)
            self._has_weights = True

        if self._has_units:
            if not all(isinstance(unit, Unit) for unit in self.units):
                raise TypeError('All units must be of type Unit')

            unit_types = {type(unit) for unit in self.units}
            if len(unit_types) != 1:
                raise TypeError('All units must be of the same Unit subclass.')

            self.unit_type = type(self.units[0])

    def add_unit(self, unit: Unit):
        if not isinstance(unit, Unit):
            raise TypeError('Unit must be of type Unit')

        if not self._has_units:
            self.units.append(unit)
            self.unit_type = type(unit)
            self._has_units = True

        elif isinstance(unit, self.unit_type):
            self.units.append(unit)
        else:
            raise TypeError(f'Units must be of type {type(self.unit_type)}, received {type(unit)} instead')

        return self

    def average_metrics(self):
        metrics = []

        if not self._has_units:
            return

        for idx, unit in enumerate(self.units):
            for evaluation_type, metrics_df in unit.metrics.items():
                df = metrics_df.with_columns([
                    pl.lit(idx).alias('Unit'),
                    pl.lit(evaluation_type).alias('Set')
                ])
                metrics.append(df)

        self.metrics = (pl.concat(metrics)
                        .group_by(['Set', 'Metric', 'Group'])
                        .agg([
                            pl.col('Value').mean().round(5).alias('Mean'),
                            pl.col('Value').median().round(5).alias('Median'),
                            pl.col('Value').std().round(5).alias('StdDev'),
                        ])
                        .sort(['Set', 'Metric', 'Group'])
        )
        return

    @abstractmethod
    def predict(self, x_array: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def score(self, x_array: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        pass

    def save(self, path: str):
        joblib.dump(self, path, compress=('xz', 5))

    @staticmethod
    def load(path: str):
        return joblib.load(path)


class ClassifierEnsemble(Ensemble):
    """
    For ClassifierEnsemble, "voting" governs only the .predict_proba method.
    Regardless of its setting, the .predict method ALWAYS returns the mode of predictions.
    For the .predict_proba method:
        - 'soft': returns the mean value
        - 'weighted': returns the weighted average
    """
    def __init__(self):
        super().__init__()
        self._task = 'Classification'

    def predict(self, x_array: np.ndarray) -> np.ndarray:
        predictions = [unit.predict(x_array).reshape(-1, 1) for unit in self.units]
        return stats.mode(np.hstack(predictions), axis=1)[0]

    def predict_proba(self, x_array: np.ndarray) -> np.ndarray:
        if self.voting == 'weighted' and self.weights is None:
            raise RuntimeWarning('Voting was set to "weighted" but self.weights were not set.'
                                 'Falling back to "soft" voting.')
            self.weights = np.ones(len(self.units)) / len(self.units)
            self.voting = 'soft'

        probs = [unit.predict_proba(x_array).reshape(-1, 1) for unit in self.units]

        if self.voting == 'soft':
            return np.mean(np.hstack(probs), axis=1)

        elif self.voting == 'weighted':
            if len(self.weights) != len(self.units):
                raise ValueError('Number of Units does not match the number of passed weights.')
            return np.average(np.hstack(probs), axis=1, weights=self.weights)
        else:
            raise ValueError('Voting must be either "soft" or "weighted"')

    def score(self, x_array: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        if not self._has_units:
            raise RuntimeError("No Units to score")

        y_pred = self.predict(x_array)
        y_score = self.predict_proba(x_array)

        cls_unit = self.units[0]

        metrics = (cls_unit.evaluate(y_true=y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = cls_unit.transformer.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:  # If for some reason a group is empty...
                    g_y_true = y_true[mask]
                    g_y_pred = y_pred[mask]
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = cls_unit.evaluate(y_true=g_y_true, y_pred=g_y_pred, sample_weight=g_sample_weight)
                    g_metrics = g_metrics.with_columns(pl.lit(group_id).alias('Group'))
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        return metrics


class RegressorEnsemble(Ensemble):
    """
    For RegressorEnsemble, "voting" gives the following behaviour:
        - 'hard': returns the median value
        - 'soft': returns the mean value
        - 'weighted': returns the weighted average
    """
    def __init__(self):
        super().__init__()
        self._task = 'Regression'

    def predict(self, x_array: np.ndarray) -> np.ndarray:
        if self.voting == 'weighted' and self.weights is None:
            raise RuntimeWarning('Voting was set to "weighted" but self.weights were not set.'
                                 'Falling back to "soft" voting.')
            self.weights = np.ones(len(self.units)) / len(self.units)
            self.voting = 'soft'

        predictions = [unit.predict(x_array).reshape(-1, 1) for unit in self.units]

        if self.voting == 'hard':
            return np.median(np.hstack(predictions), axis=1)

        elif self.voting == 'soft':
            return np.mean(np.hstack(predictions), axis=1)

        elif self.voting == 'weighted':
            if len(self.weights) != len(self.units):
                raise ValueError('Number of Units does not match the number of passed weights.')
            return np.average(np.hstack(predictions), axis=1, weights=self.weights)
        else:
            raise ValueError('Voting must be either "hard", "soft" or "weighted"')


    def score(self, x_array: np.ndarray, y_true: np.ndarray, sample_weight: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None):
        if not self._has_units:
            raise RuntimeError("No Units to score")

        y_pred = self.predict(x_array)

        reg_unit = self.units[0]

        metrics = (reg_unit.evaluate(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = reg_unit.transformer.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:  # If for some reason a group is empty...
                    g_y_true = y_true[mask]
                    g_y_pred = y_pred[mask]
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = reg_unit.evaluate(y_true=g_y_true, y_pred=g_y_pred, sample_weight=g_sample_weight)
                    g_metrics = g_metrics.with_columns(pl.lit(group_id).alias('Group'))
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        return metrics


class EnsembleFactory:
    """
    Factory class for building Ensembles from different resampling strategies.
    """

    @staticmethod
    def from_kfold():
        raise NotImplementedError

    @staticmethod
    def from_bootstrap():
        raise NotImplementedError
