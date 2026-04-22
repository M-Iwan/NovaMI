import numpy as np
import polars as pl
import pytest

from novami.data.manager import KFoldManager, TTManager


def _base_df():
    return pl.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "CCCl"],
            "X": [np.array([1.0, 0.0]), np.array([0.5, 1.0]), np.array([0.0, 1.5]), np.array([1.5, 0.5])],
            "Y": [0.1, 0.2, 0.3, 0.4],
            "Fold": [0, 1, 2, 0],
            "W": [1.0, 2.0, 1.0, 0.5],
            "G": ["A", "A", "B", "B"],
            "Set": ["Train", "Train", "Test", "Test"],
        }
    )


def test_kfold_manager_basic_accessors():
    manager = KFoldManager(
        df=_base_df(),
        smiles_col="SMILES",
        features_col="X",
        target_col="Y",
        fold_col="Fold",
        test_fold=2,
        weights_col="W",
        groups_col="G",
    )
    train = manager.get_train_data(0)
    eval_ = manager.get_eval_data(0)
    test = manager.get_test_data()
    assert train["x_array"].ndim == 2
    assert eval_["y_true"].ndim == 1
    assert len(test["y_true"]) == 1
    assert len(manager.get_test_smiles()) == 1


def test_kfold_manager_raises_for_invalid_fold_access():
    manager = KFoldManager(
        df=_base_df(),
        smiles_col="SMILES",
        features_col="X",
        target_col="Y",
        fold_col="Fold",
        test_fold=2,
    )
    with pytest.raises(ValueError):
        manager.get_train_data(2)


def test_tt_manager_basic_accessors():
    manager = TTManager(
        df=_base_df(),
        smiles_col="SMILES",
        features_col="X",
        target_col="Y",
        set_col="Set",
        weights_col="W",
        groups_col="G",
    )
    train = manager.get_train_data()
    test = manager.get_test_data()
    assert train["x_array"].shape[0] == 2
    assert test["x_array"].shape[0] == 2
    assert len(manager.get_train_smiles()) == 2
