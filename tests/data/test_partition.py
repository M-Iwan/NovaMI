import numpy as np
import polars as pl
import pytest

import novami.data.partition as part


def _base_df():
    return pl.DataFrame(
        {
            "SMILES": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF"],
            "Y": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            "Cluster": [0, 0, 1, 1, 2, 2],
            "Time": [1, 2, 3, 4, 5, 6],
        }
    )


def test_validate_features_and_targets():
    x = part.validate_features(np.array([1.0, 2.0]))
    y = part.validate_targets(np.array([[1.0], [2.0]]))
    assert x.shape == (1, 2)
    assert y.shape == (2,)


def test_validate_dataframe_accepts_polars():
    df = _base_df()
    out = part.validate_dataframe(df)
    assert isinstance(out, pl.DataFrame)


def test_random_and_stratified_train_test_split():
    df = _base_df().with_row_index("MolID")
    train_df, test_df = part.random_train_test_split(df, fraction=0.33, id_col="MolID", seed=1)
    assert train_df.height + test_df.height == df.height

    train_s, test_s = part.stratified_train_test_split(df, fraction=0.33, strat_col="Y", id_col="MolID")
    assert train_s.height + test_s.height == df.height


def test_minimal_train_test_split_and_temporal():
    df = _base_df()
    out = part.minimal_train_test_split(df, fraction=0.33, cluster_col="Cluster")
    assert "Set" in out.columns

    out_t = part.temporal_split(df, time_col="Time", fraction=0.33)
    assert "Set" in out_t.columns


def test_kfold_splitters():
    df = _base_df()
    rk = part.random_kfold_split(df, n_folds=3, seed=1)
    sk = part.stratified_kfold_split(df, strat_col="Y", n_folds=3, seed=1)
    gk = part.group_kfold_split(df, cluster_col="Cluster", n_folds=3, seed=1)
    sgk = part.stratified_group_kfold_split(df, strat_col="Y", cluster_col="Cluster", n_folds=3, seed=1)
    assert "Fold" in rk.columns and "Fold" in sk.columns and "Fold" in gk.columns and "Fold" in sgk.columns


def test_cluster_based_kfold_wrappers(monkeypatch):
    df = _base_df()

    monkeypatch.setattr(part, "murcko_cluster", lambda df, smiles_col, generic: df.with_columns(pl.Series("Cluster", [0, 0, 1, 1, 2, 2])))
    monkeypatch.setattr(part, "butina_cluster", lambda df, fp_col, threshold, batch_size, n_jobs: df.with_columns(pl.Series("Cluster", [0, 0, 1, 1, 2, 2])))
    monkeypatch.setattr(part, "cc_cluster", lambda df, features_col, threshold, metric, n_jobs: df.with_columns(pl.Series("Cluster", [0, 0, 1, 1, 2, 2])))
    monkeypatch.setattr(part, "are_folds_balanced", lambda fold_sizes, tolerance: True)

    df_fp = df.with_columns(pl.Series("Morgan", [np.array([1, 0])] * df.height))
    out_m = part.murcko_kfold_split(df, smiles_col="SMILES", n_folds=3)
    out_b = part.butina_kfold_split(df_fp, smiles_col="SMILES", fp_col="Morgan", n_folds=3, n_jobs=1)
    out_c = part.cc_kfold_split(df_fp, smiles_col="SMILES", features_col="Morgan", n_folds=3, n_jobs=1)
    assert "Fold" in out_m.columns and "Fold" in out_b.columns and "Fold" in out_c.columns


def test_collate_strat_and_fold_balance():
    df = _base_df()
    out = part.collate_strat(df, strat_col=["Y", "Cluster"])
    assert "Bin" in out.columns

    assert part.are_folds_balanced(np.array([10, 11, 9]), tolerance=0.2)
    with pytest.raises(ValueError):
        part.are_folds_balanced(np.array([-1, 2, 3]))
