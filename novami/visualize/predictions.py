import os
from abc import ABC
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, norm

from novami.data.distance import k_neighbors_distance
from novami.ml.models import Unit, Ensemble
from novami.visualize.utils import *


class Analyzer(ABC):
    """
    Class for analyzing predictions made by models.
    """

    def __init__(self, model: Union[Unit, Ensemble], df: pl.DataFrame, features_col: str, target_col: str,
                 fold_col: str, test_fold: int, weights_col: Optional[str] = None, groups_col: Optional[str] = None,
                 save_directory: Optional[str] = None):

        if not isinstance(model, (Unit, Ensemble)):
            raise TypeError(f'Model must inherit from type "Unit" or "Ensemble"')

        self.model = model
        self.task = self.model._task

        if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise TypeError(f'Expected either pandas or polars DataFrame, got {type(df)} instead')

        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        self.df = df

        required_cols = [features_col, target_col, fold_col]
        if weights_col is not None:
            required_cols.append(weights_col)
        if groups_col is not None:
            required_cols.append(groups_col)

        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f'Column {col} not found in DataFrame')

        self.features_col = features_col
        self.target_col = target_col
        self.fold_col = fold_col
        self.weights_col = weights_col
        self.groups_col = groups_col
        self._has_weights = True if self.weights_col is not None else False
        self._has_groups = True if self.groups_col is not None else False

        if test_fold not in (folds := self.df[self.fold_col].unique()):
            raise ValueError(f'Test fold {test_fold} not found in DataFrame. Available folds: {folds} ')

        self.test_fold = test_fold

        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)

        features = np.vstack(self.df[self.features_col].to_numpy())
        self.df = (self.df
            .with_columns([
                pl.Series('y_pred', self.model.predict(features)),
                pl.when(pl.col(self.fold_col) == test_fold)
                  .then(pl.lit('Test'))
                  .otherwise(pl.lit('Train'))
                  .alias('Set')

            ])
            .with_columns(
                (pl.col(self.target_col) - pl.col('y_pred')).alias('Residual Error')
            )
        )

        self.train_df = self.df.filter(pl.col(self.fold_col) != self.test_fold)
        self.test_df = self.df.filter(pl.col(self.fold_col) == self.test_fold)

    @staticmethod
    def try_set_font():
        try:
            set_font('arial.ttf')
        except FileNotFoundError:
            print(f'Arial font not found in matplotlib.')

class ClassifierAnalyzer(Analyzer):
    def __init__(self, model: Union[Unit, Ensemble], df: pl.DataFrame, features_col: str, target_col: str,
                 fold_col: str, test_fold: int, weights_col: Optional[str] = None, groups_col: Optional[str] = None,
                 save_directory: Optional[str] = None):

        super().__init__(model=model, df=df, features_col=features_col, target_col=target_col,
                         fold_col=fold_col, test_fold=test_fold, weights_col=weights_col, groups_col=groups_col,
                         save_directory=save_directory)

        self.df = self.df.with_columns(
                pl.Series('y_score', self.model.predict_proba(self.df[self.features_col].to_numpy()))
            )


class RegressorAnalyzer(Analyzer):
    def __init__(self, model: Union[Unit, Ensemble], df: pl.DataFrame, features_col: str, target_col: str,
                 fold_col: str, test_fold: int, weights_col: Optional[str] = None, groups_col: Optional[str] = None):

        super().__init__(model=model, df=df, features_col=features_col, target_col=target_col,
                         fold_col=fold_col, test_fold=test_fold, weights_col=weights_col, groups_col=groups_col)

    def predicted_vs_observed(self, name: Optional[str] = None, plot_kwargs: Optional[dict] = None):
        """
        Plot predicted vs observed values.

        Parameters
        ----------
        name: Optional[str]
            If not None, the name of file for saving.
        plot_kwargs: dict
            Other parameters passed to sns.scatterplot.
        """

        self.try_set_font()
        sns.set_style('whitegrid')
        sns.set_context('paper')
        tpp = two_point_palette()

        dparams = {}
        if plot_kwargs is not None:
            dparams.update(plot_kwargs)

        g = sns.relplot(self.df, x=self.target_col, y='y_pred', col='Set', col_order=['Train', 'Test'],
                        color=tpp[0], facet_kws={'sharex': False, 'sharey': False}, **dparams)

        x_train, y_train = self.get_line(self.train_df[self.target_col].to_numpy())
        x_test, y_test = self.get_line(self.test_df[self.target_col].to_numpy())

        sns.lineplot(x=x_train, y=y_train, ax=g.figure.axes[0], color=tpp[1])
        sns.lineplot(x=x_test, y=y_test, ax=g.figure.axes[1], color=tpp[1])

        g.set_xlabels('Observed Value')
        g.set_ylabels('Predicted Value')

        if name is not None:
            save_path = os.path.join(self.save_directory, f'{name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def predicted_vs_residuals(self, name: Optional[str] = None, plot_kwargs: Optional[dict] = None):
        """
        Plot predicted values vs their residuals to detect heteroscedasticity.

        Parameters
        ----------
        name: Optional[str]
            If not None, the name of file for saving.
        plot_kwargs: dict
            Other parameters passed to sns.scatterplot.
        """
        self.try_set_font()
        sns.set_style('whitegrid')
        sns.set_context('paper')
        tpp = two_point_palette()

        dparams = {}
        if plot_kwargs is not None:
            dparams.update(plot_kwargs)


        g = sns.relplot(self.df, x='y_pred', y='Residual Error', col='Set', col_order=['Train', 'Test'],
                        color=tpp[0], facet_kws={'sharex': False, 'sharey': False}, **dparams)

        x_train, y_train = self.get_line(self.train_df[self.target_col].to_numpy(),0, 0)
        x_test, y_test = self.get_line(self.test_df[self.target_col].to_numpy(), 0, 0)

        sns.lineplot(x=x_train, y=y_train, ax=g.figure.axes[0], color=tpp[1])
        sns.lineplot(x=x_test, y=y_test, ax=g.figure.axes[1], color=tpp[1])

        g.set_xlabels('Predicted Value')
        g.set_ylabels('Residual Error')

        if name is not None:
            save_path = os.path.join(self.save_directory, f'{name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def residuals_hist(self, name: Optional[str] = None, plot_kwargs: Optional[dict] = None):
        """
        Plot histogram of residuals.

        Parameters
        ----------
        name: Optional[str]
            If not None, the name of file for saving.
        plot_kwargs: dict
            Other parameters passed to sns.histplot.
        """

        self.try_set_font()
        sns.set_style('white')
        sns.set_context('paper')
        tpp = two_point_palette()

        dparams = {'bins': 25}
        if plot_kwargs is not None:
            dparams.update(plot_kwargs)

        g = sns.displot(self.df, x='Residual Error', col='Set', col_order=['Train', 'Test'], stat='density',
                        color=tpp[0], kind='hist', facet_kws={'sharex': False, 'sharey': False}, **dparams)

        x_train, y_train = self.get_normal(self.train_df['Residual Error'].to_numpy())
        x_test, y_test = self.get_normal(self.test_df['Residual Error'].to_numpy())

        sns.lineplot(x=x_train, y=y_train, ax=g.figure.axes[0], color=tpp[1])
        sns.lineplot(x=x_test, y=y_test, ax=g.figure.axes[1], color=tpp[1])

        g.set_xlabels('Residual Error')

        if name is not None:
            save_path = os.path.join(self.save_directory, f'{name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def qq_plot(self, name: Optional[str] = None, plot_kwargs: Optional[dict] = None):
        """
        Make a QQ plot from residuals on the test fold.

        Parameters
        ----------
        name: Optional[str]
            If not None, the name of file for saving.
        plot_kwargs: dict
            Other parameters passed to sns.scatterplot.
        """

        self.try_set_font()
        sns.set_style('whitegrid')
        sns.set_context('paper')
        tpp = two_point_palette()

        dparams = {'alpha': 0.8}
        if plot_kwargs is not None:
            dparams.update(plot_kwargs)

        residuals = self.train_df['Residual Error'].to_numpy()
        (quantiles, y_values), (slope, intercept, r) = probplot(residuals)
        x_train, y_train = self.get_line(quantiles, slope, intercept)
        train_df = pl.DataFrame({
            'X': quantiles,
            'Y': y_values,
            'Set': 'Train'
        })

        residuals = self.test_df['Residual Error'].to_numpy()
        (quantiles, y_values), (slope, intercept, r) = probplot(residuals)
        x_test, y_test = self.get_line(quantiles, slope, intercept)
        test_df = pl.DataFrame({
            'X': quantiles,
            'Y': y_values,
            'Set': 'Test'
        })

        plot_df = pl.concat([train_df, test_df])

        g = sns.relplot(plot_df, x='X', y='Y', col='Set', col_order=['Train', 'Test'],
                        color=tpp[0], facet_kws={'sharex': False, 'sharey': False}, **dparams)

        sns.lineplot(x=x_train, y=y_train, ax=g.figure.axes[0], color=tpp[1])
        sns.lineplot(x=x_test, y=y_test, ax=g.figure.axes[1], color=tpp[1])

        g.set_xlabels('Theoretical Quantiles')
        g.set_ylabels('Observed Quantiles')

        if name is not None:
            save_path = os.path.join(self.save_directory, f'{name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def residual_vs_distance(self, metric: str, nearest_k: Optional[List[int]] = None, n_jobs: Optional[int] = 1,
                             name: Optional[str] = None, plot_kwargs: Optional[dict] = None):
        """
        TODO: To be used with novami.data.similarity k_neighbors_distance.
        """
        self.try_set_font()
        sns.set_style('whitegrid')
        sns.set_context('paper')
        tpp = two_point_palette()

        dparams = {'alpha': 0.6, 'size': 1}
        if plot_kwargs is not None:
            dparams.update(plot_kwargs)

        if nearest_k is None:
            nearest_k = [1, 3, 5]

        array_1 = np.vstack(self.train_df[self.features_col].to_numpy())
        array_2 = np.vstack(self.test_df[self.features_col].to_numpy())

        neighbor_df = k_neighbors_distance(
            array_1=array_1,
            array_2=array_2,
            metric=metric,
            n_jobs=n_jobs,
            nearest_k=nearest_k,
            furthest_k=None)

        errors_df = (self.test_df[['Residual Error']]
                    .with_columns(
                        pl.col('Residual Error').abs().alias('Absolute Error')
                    )
                    .drop('Residual Error'))


        plot_df = (pl.concat([errors_df, neighbor_df], how='horizontal')
                   .unpivot(index='Absolute Error', variable_name='Aggregation', value_name='Distance')
                   .filter(pl.col('Aggregation') != 'Max')
        )

        g = sns.relplot(plot_df, x='Distance', y='Absolute Error', col='Aggregation', col_wrap=2, color=tpp[0],
                        facet_kws={'sharex': False, 'sharey': False}, **dparams)

        for ax in g.axes.flatten():
            aggregation = ax.title.get_text().split('=')[-1].strip()
            sub_df = plot_df.filter(pl.col('Aggregation') == aggregation)
            sns.regplot(sub_df, x='Distance', y='Absolute Error', scatter=False, ax=ax, color=tpp[1])

        if name is not None:
            save_path = os.path.join(self.save_directory, f'{name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def get_line(values: np.ndarray, slope: float = 1, intercept: float = 0):
        x = np.linspace(min(values), max(values), 100)
        y = x * slope + intercept
        return x, y

    @staticmethod
    def get_normal(values: np.ndarray):
        mean = np.mean(values)
        std = np.std(values)

        x = np.linspace(min(values), max(values), 100)
        y = norm.pdf(x, mean, std)
        return x, y
