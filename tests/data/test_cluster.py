import numpy as np
import pandas as pd
import polars as pl

import novami.data.cluster as cluster


def test_butina_cluster_assigns_cluster_column_with_mocked_similarity(monkeypatch):
    df = pl.DataFrame(
        {
            "Morgan": [np.array([1, 0]), np.array([1, 0]), np.array([0, 1])],
        }
    )

    monkeypatch.setattr(cluster, "embeddings_to_rdkit", lambda arr: list(range(len(arr))))

    def fake_bulk_tanimoto_similarity(query, rest):
        return [1.0 if query == item else 0.0 for item in rest]

    monkeypatch.setattr(cluster.DataStructs, "BulkTanimotoSimilarity", fake_bulk_tanimoto_similarity)
    out = cluster.butina_cluster(df, fp_col="Morgan", threshold=0.1, n_jobs=1, batch_size=2)
    assert "Cluster" in out.columns
    assert out.height == 3


def test_murcko_cluster_handles_invalid_smiles():
    df = pl.DataFrame({"SMILES": ["CCO", "not-a-smiles"]})
    out = cluster.murcko_cluster(df, smiles_col="SMILES", generic=False)
    assert "Cluster" in out.columns
    assert "InvalidMolecule" in out["Cluster"].to_list()


def test_cc_cluster_with_mocked_distance_matrix(monkeypatch):
    df = pd.DataFrame(
        {
            "Morgan": [np.array([1, 0]), np.array([1, 0]), np.array([0, 1])],
        }
    )
    fake_dist = np.array(
        [
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.9],
            [0.8, 0.9, 0.0],
        ]
    )
    monkeypatch.setattr(cluster, "distance_matrix", lambda array_1, array_2, metric, n_jobs: fake_dist)
    out = cluster.cc_cluster(df, features_col="Morgan", threshold=0.2, metric="jaccard", n_jobs=1)
    assert "Cluster" in out.columns
    assert len(set(out["Cluster"])) == 2
