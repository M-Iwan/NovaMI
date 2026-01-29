import os
import pickle
from collections import defaultdict
from typing import ClassVar, Dict, List
import numpy as np
import pandas as pd
import torch
import captum
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.special import expit
from tqdm import tqdm
from novami.data.manipulate import remove_zero_var_df, remove_inf_df, remove_nan_df
from projects.drid.eval import DatasetManager


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

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Environment: snellius
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

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Environment: snellius
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


class XIterEnsemble(torch.nn.Module):
    """
    PyTorch wrapper for IterEnsemble to make it compatible with Captum attribution methods.

    Parameters
    ----------
    ensemble : IterEnsemble
        The ensemble model to wrap for attribution analysis.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@bayer.com / mateusz.iwan@hotmail.com
    Environment: snellius
    """

    def __init__(self, ensemble: IterEnsemble):
        super(XIterEnsemble, self).__init__()
        self.ensemble = ensemble

    def forward(self, x_array: torch.FloatTensor, x_demo: torch.LongTensor):

        if x_array.ndim == 1 or x_array.shape[0] == 1:
            x_array = x_array.reshape(1, -1)
        if x_demo.ndim == 1 or x_demo.shape[0] == 1:
            x_demo = x_demo.reshape(1, -1)

        x_array = x_array.detach().cpu().numpy()
        x_demo = x_demo.detach().cpu().numpy()

        probs = self.ensemble.predict_proba(x_array, x_demo)
        return torch.FloatTensor(probs)

def normalize_attribution(attrs: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def entry_attribution(explainer: captum.attr.KernelShap, x_array: np.ndarray, x_demo: np.ndarray,
                      n_samples:int = 256) -> dict:
    """
    Calculate feature attributions for a single input using KernelSHAP.

    Computes attributions for both molecular descriptors and demographic features
    by holding one constant while varying the other. Also calculates their relative importance.

    Parameters
    ----------
    explainer : captum.attr.KernelShap
        Initialized KernelShap explainer wrapping the model.
    x_array : np.ndarray
        Molecular descriptor array for a single sample.
    x_demo : np.ndarray
        Demographic feature array for a single sample.
    n_samples : int, optional
        Number of samples to use for KernelSHAP approximation, by default 256.

    Returns
    -------
    dict
        Dictionary containing attribution results.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@bayer.com / mateusz.iwan@hotmail.com
    Environment: snellius
    """

    if x_array.ndim == 1 or x_array.shape[0] == 1:
        x_array = x_array.reshape(1, -1)
    if x_demo.ndim == 1 or x_demo.shape[0] == 1:
        x_demo = x_demo.reshape(1, -1)

    t_array = torch.from_numpy(x_array)
    t_demo = torch.from_numpy(x_demo)

    t_array_zero = torch.zeros_like(t_array)  # use the same demographic representation
    t_demo_zero = torch.zeros_like(t_demo)  # use the same molecular representation

    # keep the demographic part constant - evaluate the effects of molecular representation
    t_attr_with_zeroed_mol = explainer.attribute(
        inputs=(t_array, t_demo),
        baselines=(t_array_zero, t_demo),
        n_samples=n_samples
    )
    mol_attr_with_zeroed_mol = t_attr_with_zeroed_mol[0].detach().cpu().numpy()
    demo_attr_with_zeroed_mol = t_attr_with_zeroed_mol[1].detach().cpu().numpy()

    # keep the molecular part constant - evaluate the effects of demographics
    t_attr_with_zeroed_demo = explainer.attribute(
        inputs=(t_array, t_demo),
        baselines=(t_array, t_demo_zero),
        n_samples=n_samples
    )
    mol_attr_with_zeroed_demo = t_attr_with_zeroed_demo[0].detach().cpu().numpy()
    demo_attr_with_zeroed_demo = t_attr_with_zeroed_demo[1].detach().cpu().numpy()

    t_attr_with_both_zeroed = explainer.attribute(
        inputs=(t_array, t_demo),
        baselines=(t_array_zero, t_demo_zero),
        n_samples=n_samples
    )
    mol_attr_with_both_zeroed = t_attr_with_both_zeroed[0].detach().cpu().numpy()
    demo_attr_with_both_zeroed = t_attr_with_both_zeroed[1].detach().cpu().numpy()

    results = {
        'mol_attr_with_zeroed_mol': mol_attr_with_zeroed_mol,
        'mol_attr_with_zeroed_demo': mol_attr_with_zeroed_demo,
        'mol_attr_with_both_zeroed': mol_attr_with_both_zeroed,
        'demo_attr_with_zeroed_mol': demo_attr_with_zeroed_mol,
        'demo_attr_with_zeroed_demo': demo_attr_with_zeroed_demo,
        'demo_attr_with_both_zeroed': demo_attr_with_both_zeroed
    }

    return results


def fold_attribution(df: pd.DataFrame, ensemble: IterEnsemble, desc_col: str, demo_col: str, test_fold: int,
                     n_samples: int, save_dir: str) -> dict:
    """
    Calculate and save attributions for all samples in a specific test fold.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing test samples for the fold.
    ensemble: IterEnsemble
        Trained instance of IterEnsemble.
    desc_col : str
        Name of the column containing molecular descriptors.
    demo_col : str
        Name of the column containing demographic features.
    test_fold : int
        Index of the test fold being processed.
    n_samples: int
        Number of samples to use for KernelSHAP approximation.
    save_dir : str
        Directory to save attribution results.

    Returns
    -------
    dict
        Dictionary containing attribution arrays for the fold.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@bayer.com / mateusz.iwan@hotmail.com
    Environment: snellius
    """

    xEnsemble = XIterEnsemble(ensemble)
    explainer = captum.attr.KernelShap(xEnsemble)

    x_arrays = df[desc_col].to_numpy()
    x_demos = df[demo_col].to_numpy()

    fold_results = []

    for x_array, x_demo in tqdm(zip(x_arrays, x_demos), total=len(x_arrays)):
        out = entry_attribution(
            explainer=explainer,
            x_array=x_array,
            x_demo=x_demo,
            n_samples=n_samples
        )
        fold_results.append(out)

    fold_results = {
        'mol_attrs_with_zeroed_mol': np.vstack([res['mol_attr_with_zeroed_mol'] for res in fold_results]),
        'mol_attrs_with_zeroed_demo': np.vstack([res['mol_attr_with_zeroed_demo'] for res in fold_results]),
        'mol_attrs_with_both_zeroed': np.vstack([res['mol_attr_with_both_zeroed'] for res in fold_results]),
        'demo_attrs_with_zeroed_mol': np.vstack([res['demo_attr_with_zeroed_mol'] for res in fold_results]),
        'demo_attrs_with_zeroed_demo': np.vstack([res['demo_attr_with_zeroed_demo'] for res in fold_results]),
        'demo_attrs_with_both_zeroed': np.vstack([res['demo_attr_with_both_zeroed'] for res in fold_results]),
        'test_fold': test_fold
    }

    save_path = os.path.join(save_dir, f'xai_tf_{test_fold}.pkl')
    pickle.dump(fold_results, open(save_path, 'wb'))

    return fold_results


def summarise_attributions(attrs: np.ndarray):
    """
    Calculate summary statistics for attribution values.

    Computes mean, standard deviation, median, confidence intervals, and absolute mean
    for attribution values across samples.

    Parameters
    ----------
    attrs : np.ndarray
        Array of attribution values, shape (n_samples, n_features).

    Returns
    -------
    dict
        Dictionary containing statistics:
        - 'mean': Mean attribution value for each feature
        - 'std': Standard deviation of attribution for each feature
        - 'median': Median attribution for each feature
        - 'ci_low': Lower bound of 95% confidence interval
        - 'ci_high': Upper bound of 95% confidence interval
        - 'abs_mean': Mean of absolute attribution values

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@bayer.com / mateusz.iwan@hotmail.com
    Environment: snellius
    """

    stats = {
        'mean': np.mean(attrs, axis=0),
        'std': np.std(attrs, axis=0),
        'median': np.median(attrs, axis=0),
        'ci_low': np.percentile(attrs, 2.5, axis=0),
        'ci_high': np.percentile(attrs, 97.5, axis=0),
        'abs_mean': np.mean(np.abs(attrs), axis=0)
    }
    return stats


def data_attribution(dataset_path: str, desc_path: str, ensemble_dir: str, n_samples: int, save_dir: str):
    """
    Perform attribution analysis across all folds of a dataset.

    Loads dataset and descriptors, processes each fold, computes attributions for all samples, and aggregates statistics across folds.

    Parameters
    ----------
    dataset_path : str
        Path to the pickled dataset file.
    desc_path : str
        Path to the pickled descriptors file.
    ensemble_dir : str
        Directory containing ensemble model files.
    n_samples: int
        Number of samples to use for KernelSHAP approximation.
    save_dir : str
        Directory to save attribution results and summaries.

    Returns
    -------
    dict
        Dictionary containing summarized attribution results.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@bayer.com / mateusz.iwan@hotmail.com
    Environment: snellius
    """

    dataset = pickle.load(open(dataset_path, 'rb'))
    descriptors = pickle.load(open(desc_path, 'rb'))

    desc_col = ensemble_dir.rstrip('/').split('/')[-1]

    descriptors = remove_inf_df(descriptors, desc_col)
    descriptors = remove_nan_df(descriptors, desc_col)
    descriptors = remove_zero_var_df(descriptors, desc_col)

    dataset = dataset.merge(descriptors[['SMILES', desc_col]], on='SMILES', how='inner')

    os.makedirs(save_dir, exist_ok=True)

    data_results = []

    for test_fold in dataset['Fold'].unique():
        train_df = dataset[dataset['Fold'] != test_fold]

        # It initializes the objects for FoldUnit... And I was so proud of this code when I first wrote it...
        train_mgr = DatasetManager(
            df=train_df,
            desc_col=desc_col,
            sign_names=['Sex', 'Age', 'Weight']
        )
        FoldUnit.selectors = train_mgr.selectors
        FoldUnit.scalers = train_mgr.scalers
        FoldUnit.train_idxs = train_mgr.train_idxs
        FoldUnit.eval_idxs = train_mgr.eval_idxs

        test_df = dataset[dataset['Fold'] == test_fold].reset_index(drop=True)

        ensemble_path = os.path.join(ensemble_dir, f'ensemble_tf_{test_fold}.pkl')
        ensemble = pickle.load(open(ensemble_path, 'rb'))

        fold_results = fold_attribution(
            df=test_df,
            ensemble=ensemble,
            desc_col=desc_col,
            demo_col='DemoFP',
            test_fold=test_fold,
            n_samples=n_samples,
            save_dir=save_dir
        )

        data_results.append(fold_results)

    mol_attrs_with_zeroed_mol = np.vstack([res['mol_attrs_with_zeroed_mol'] for res in data_results])
    mol_attrs_with_zeroed_demo = np.vstack([res['mol_attrs_with_zeroed_demo'] for res in data_results])
    mol_attrs_with_both_zeroed = np.vstack([res['mol_attrs_with_both_zeroed'] for res in data_results])
    demo_attrs_with_zeroed_mol = np.vstack([res['demo_attrs_with_zeroed_mol'] for res in data_results])
    demo_attrs_with_zeroed_demo = np.vstack([res['demo_attrs_with_zeroed_demo'] for res in data_results])
    demo_attrs_with_both_zeroed = np.vstack([res['demo_attrs_with_both_zeroed'] for res in data_results])

    # General stats
    mol_stats_with_zeroed_mol = summarise_attributions(mol_attrs_with_zeroed_mol)
    mol_stats_with_zeroed_demo = summarise_attributions(mol_attrs_with_zeroed_demo)
    mol_stats_with_both_zeroed = summarise_attributions(mol_attrs_with_both_zeroed)
    demo_stats_with_zeroed_mol = summarise_attributions(demo_attrs_with_zeroed_mol)
    demo_stats_with_zeroed_demo = summarise_attributions(demo_attrs_with_zeroed_demo)
    demo_stats_with_both_zeroed = summarise_attributions(demo_attrs_with_both_zeroed)

    # Relative importance from both zeroed variant
    mol_importance = mol_stats_with_both_zeroed['abs_mean'].mean()
    demo_importance = demo_stats_with_both_zeroed['abs_mean'].mean()

    if mol_importance > 0 and demo_importance > 0:
        relative_mol_importance = mol_importance / (total_importance := mol_importance + demo_importance)
        relative_demo_importance = demo_importance / total_importance
    else:
        print('Something went no yes')
        relative_mol_importance = np.nan
        relative_demo_importance = np.nan

    # Demographic features processing
    demo_features = ['Sex_Male', 'Sex_Female', 'Age_Children', 'Age_Adolescent', 'Age_Adult', 'Age_Elderly',
                     'Weight_Low', 'Weight_Average', 'Weight_High']

    dfs = []

    for idx, name in enumerate(demo_features):
        sub_df_with_zeroed_demo = pd.DataFrame({
            'Feature': [name],
            'Means': demo_stats_with_zeroed_demo['mean'][idx],
            'Stds': demo_stats_with_zeroed_demo['std'][idx],
            'Medians': demo_stats_with_zeroed_demo['median'][idx],
            'CI_Low': demo_stats_with_zeroed_demo['ci_low'][idx],
            'CI_High': demo_stats_with_zeroed_demo['ci_high'][idx],
            'Abs_Means': demo_stats_with_zeroed_demo['abs_mean'][idx],
            'Zeroed': 'Demo'
        })
        dfs.append(sub_df_with_zeroed_demo)

        sub_df_with_both_zeroed = pd.DataFrame({
            'Feature': [name],
            'Means': demo_stats_with_both_zeroed['mean'][idx],
            'Stds': demo_stats_with_both_zeroed['std'][idx],
            'Medians': demo_stats_with_both_zeroed['median'][idx],
            'CI_Low': demo_stats_with_both_zeroed['ci_low'][idx],
            'CI_High': demo_stats_with_both_zeroed['ci_high'][idx],
            'Abs_Means': demo_stats_with_both_zeroed['abs_mean'][idx],
            'Zeroed': 'Both'
        })
        dfs.append(sub_df_with_both_zeroed)

    demo_df = pd.concat(dfs).reset_index(drop=True)

    # Summary
    summary = {
        'mol_stats_with_zeroed_mol'  : mol_stats_with_zeroed_mol,
        'mol_stats_with_zeroed_demo' : mol_stats_with_zeroed_demo,
        'mol_stats_with_both_zeroed' : mol_stats_with_both_zeroed,
        'demo_stats_with_zeroed_mol' : demo_stats_with_zeroed_mol,
        'demo_stats_with_zeroed_demo': demo_stats_with_zeroed_demo,
        'demo_stats_with_both_zeroed': demo_stats_with_both_zeroed,
        'relative_mol_importance' : relative_mol_importance,
        'relative_demo_importance' : relative_demo_importance,
        'demo_df': demo_df
    }

    summary_path = os.path.join(save_dir, f'xai_summary.pkl')
    pickle.dump(summary, open(summary_path, 'wb'))

    return summary
