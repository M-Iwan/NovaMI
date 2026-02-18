from copy import deepcopy

from novami.data.manager import TTManager, KFoldManager
from novami.ml.models import *
from novami.ml.utils import prepare_unit


def tt_evaluate(model_name: str, df: pl.DataFrame, smiles_col: str, features_col: str, target_col: str,
                set_col: str = "Set", weights_col: str = None, groups_col: str = None, n_jobs: int = 1,
                model_params: dict = None, task:str = 'regression'):
    """
    Evaluate a model using train-test split.

    Parameters
    ----------
    model_name: str
        Name of the model to use. Must be included in the novami.ml.utils:get_model function
    df: pl.DataFrame
        A Polars DataFrame with features, target values, and column denoting set membership.
    smiles_col: str
        A name of a column with SMILES strings.
    features_col: str
        A name of a column with feature used for model training.
    target_col: str
        A name of a column with target values used for model training.
    set_col: str
        A name of a column with set memberships denoting Training/Test.
    weights_col: str, optional
        A name of a column with sample weights.
    groups_col: str, optional
        A name of a column with group memberships. Each group will be separately scored.
    n_jobs: int, optional
        Number of CPUs to use for model training.
    model_params: dict, optional
        Hyperparameters to use for model training. If not provided, model defaults are used.
    task: str, optional
        Either regression or classification.

    Returns
    -------
    unit: Unit
        A trained Unit instance.
    """

    if model_params is None:
        model_params = {}

    unit = prepare_unit(
        model_name=model_name,
        model_params=model_params,
        features=features_col,
        task=task,
        n_jobs=n_jobs
    )

    manager = TTManager(
        df=df,
        smiles_col=smiles_col,
        features_col=features_col,
        target_col=target_col,
        set_col=set_col,
        weights_col=weights_col,
        groups_col=groups_col
    )

    train_data = manager.get_train_data()
    test_data = manager.get_test_data()

    unit.fit(**train_data)
    unit.metrics['Training'] = unit.score(**train_data)
    unit.metrics['Testing'] = unit.score(**test_data)

    return unit


def kf_evaluate(model_name: str, df: pl.DataFrame, smiles_col: str, features_col: str, target_col: str,
                fold_col: str = "Fold", weights_col: str = None, groups_col: str = None, n_jobs: int = 1,
                model_params: dict = None, task: str = 'regression'):
    """
    Evaluate a model using KFold split.

    Parameters
    ----------
    model_name: str
        Name of the model to use. Must be included in the novami.ml.utils:get_model function
    df: pl.DataFrame
        A Polars DataFrame with features, target values, and column denoting set membership.
    smiles_col: str
        A name of a column with SMILES strings.
    features_col: str
        A name of a column with feature used for model training.
    target_col: str
        A name of a column with target values used for model training.
    fold_col: str
        A name of a column with fold memberships.
    weights_col: str, optional
        A name of a column with sample weights.
    groups_col: str, optional
        A name of a column with group memberships. Each group will be separately scored.
    n_jobs: int, optional
        Number of CPUs to use for model training.
    model_params: dict, optional
        Hyperparameters to use for model training. If not provided, model defaults are used.
    task: str, optional
        Either regression or classification.

    Returns
    -------
    ensemble: Ensemble
        A trained Ensemble instance.
    """

    if model_params is None:
        model_params = {}

    unit = prepare_unit(
        model_name=model_name,
        model_params=model_params,
        features=features_col,
        task=task,
        n_jobs=n_jobs
    )

    ensemble = {'regression': RegressorEnsemble, 'classification': ClassifierEnsemble}.get(task)()

    for test_fold in df[fold_col].unique():

        manager = KFoldManager(
            df=df,
            smiles_col=smiles_col,
            features_col=features_col,
            target_col=target_col,
            fold_col=fold_col,
            test_fold=test_fold,
            weights_col=weights_col,
            groups_col=groups_col
        )

        unit_copy = deepcopy(unit)

        train_data = manager.get_non_test_data()
        test_data = manager.get_test_data()

        unit_copy.fit(**train_data)
        unit_copy.metrics['Training'] = unit_copy.score(**train_data)
        unit_copy.metrics['Testing'] = unit_copy.score(**test_data)

        ensemble.add_unit(unit_copy)

    ensemble.average_metrics()
    return ensemble
