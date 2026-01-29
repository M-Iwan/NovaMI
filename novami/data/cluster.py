"""
TODO: clustering: dbscan, kmean, hdbscan, optics
"""
from typing import Union
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import polars as pl
from rdkit import DataStructs, Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from novami.data.manipulate import embeddings_to_rdkit
from novami.data.similarity import distance_matrix


def butina_cluster(df: Union[pd.DataFrame, pl.DataFrame], fp_col: str = 'Morgan', threshold: float = 0.3,
                   n_jobs: int = -2, batch_size: int = 256, id_col: str = 'MolID') -> pd.DataFrame:
    """
    TODO: extend to any metric type
    Performs Butina Clustering using RDKit. Parallel computing and batch processing is supported.

    Parameters
    ----------
    df: Union[pl.DataFrame, pd.DataFrame]
        Polars or Pandas DataFrame with fingerprint column.
    fp_col: str
        Name of the column with fingerprints. Default is Morgan
    threshold: float
        Distance threshold. Pair of FPs is marked as neighbors if distance <= threshold. Default is 0.3
    n_jobs: int
        Number of cores to use. Default is -2 (i.e. use all but one cores)
    batch_size: int
        Number of rows allocated to each process during neighborhood matrix calculation.
    id_col: str
        Name of the column with unique row identifiers. Default is MolID

    Returns
    -------
    df: pl.DataFrame
        Polars DataFrame with assigned neighbors and cluster_id columns.

    Notes
    -----
    Suggested starting thresholds:
    - between 0.3 and 0.4
    """

    if is_pandas := isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if id_col not in df.columns:
        df = df.with_row_index(name=id_col)

    n_samples = len(df)
    seen = np.zeros(n_samples, dtype=bool)
    cluster_ids = np.full(n_samples, -1, dtype=int)

    bit_vectors = embeddings_to_rdkit(df[fp_col].to_numpy())

    batch_indices = list(range(0, n_samples - 1, batch_size)) + [n_samples - 1]

    def compute_neighbors_batch(start_idx, end_idx, _bit_vectors, _threshold):

        distances = []

        for i in range(start_idx, end_idx):
            row_similarities = DataStructs.BulkTanimotoSimilarity(_bit_vectors[i], _bit_vectors[i + 1:])
            distances.append((1 - np.array(row_similarities)) <= _threshold)

        return distances

    # Compute neighbors in batches
    neighbor_batches = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(compute_neighbors_batch)(batch_indices[i], min(batch_indices[i + 1], n_samples - 1), bit_vectors, threshold)
        for i in range(len(batch_indices) - 1)
    )

    # Combine results into the neighbor mask
    neighbor_mask = np.zeros((n_samples, n_samples), dtype=bool)
    row_start = 0

    for batch in neighbor_batches:
        for i, row in enumerate(batch):
            row_len = len(row)

            # might be improvable using triu matrix instead
            neighbor_mask[row_start + i, row_start + i + 1: row_start + i + 1 + row_len] = row
            neighbor_mask[row_start + i + 1: row_start + i + 1 + row_len, row_start + i] = row  # Symmetric
        row_start += len(batch)

    df = df.with_columns(pl.Series('neighbors', np.sum(neighbor_mask, axis=1)))

    current_cluster_id = 0

    while not np.all(seen):

        unassigned_indices = np.where(~seen)[0]

        if df[unassigned_indices, 'neighbors'].sum() == 0:

            for idx in unassigned_indices:
                cluster_ids[idx] = current_cluster_id
                seen[idx] = True
                current_cluster_id += 1
            break

        most_neighbors_idx = unassigned_indices[np.argmax(df[unassigned_indices, 'neighbors'])]

        cluster_members = np.where(neighbor_mask[most_neighbors_idx] & ~seen)[0]
        cluster_members = np.append(cluster_members, most_neighbors_idx)

        cluster_ids[cluster_members] = current_cluster_id
        seen[cluster_members] = True

        neighbor_mask[cluster_members, :] = False
        neighbor_mask[:, cluster_members] = False

        df[np.where(seen == False)[0], 'neighbors'] = np.sum(neighbor_mask[~seen][:, ~seen], axis=1)

        current_cluster_id += 1

    df = df.with_columns(pl.Series('Cluster', cluster_ids)).drop('neighbors')

    if is_pandas:
        df = df.to_pandas()

    return df


def murcko_cluster(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', generic: bool = False):
    """
    Cluster molecules based on their (generic) Murcko Scaffolds. Molecules for which a scaffold cannot be generated
    are assigned to a single cluster.

    Parameters
    ----------
    df: Union[pd.DataFrame, pl.DataFrame]
        A pandas or polars DataFrame
    smiles_col: str
        A name of the column with SMILES strings.
    generic: bool
        A flag to make Murcko scaffolds generic, i.e. only carbon atoms and single bonds

    Returns
    -------
    df: Union[pd.DataFrame, pl.DataFrame]
    """

    def smiles_2_scaffold(smiles: str) -> str:
        """
        Calculate (generic) Murcko Scaffold from SMILES

        Parameters
        ----------
        smiles: str
         A SMILES string

        Returns
        -------
        str
        """

        nonlocal generic
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 'InvalidMolecule'
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if generic:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            return scaffold_smiles
        except Exception as e:
            print(f"Exception during calculating the scaffold of < {smiles} >\n{e}")
            return "ScaffoldNotGenerated"

    if is_pandas := isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    df = df.with_columns(pl.col(smiles_col).map_elements(
        smiles_2_scaffold, return_dtype=pl.String).alias('Cluster'))

    if is_pandas:
        df = df.to_pandas()

    return df


def cc_cluster(df: Union[pd.DataFrame, pl.DataFrame], features_col: str = 'Morgan', metric: str = 'jaccard',
               threshold: float = 0.3, n_jobs: int = 4):
    """
    Cluster molecules using connected components graphs.

    Parameters
    ----------

    df: Union[pd.DataFrame, pl.DataFrame]
        A pandas or polars DataFrame
    features_col: str
        Name of the column with fingerprints. Default is Morgan
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    threshold: float
        Distance threshold. Edge is made if distance <= threshold. Default is 0.3
    n_jobs: int
        Number of cores to use. Default is 1 (i.e. use one core)

    Returns
    -------
    df: Union[pd.DataFrame, pl.DataFrame]
    """

    if is_pandas := isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    array = np.vstack(df[features_col].to_numpy())
    dist_matrix = distance_matrix(array_1=array, array_2=array, metric=metric, n_jobs=n_jobs)
    adj_sparse = csr_matrix((dist_matrix <= threshold).astype(int))
    n_components, labels = connected_components(adj_sparse, directed=False, return_labels=True)

    df = df.with_columns(pl.Series('Cluster', labels))

    if is_pandas:
        df = df.to_pandas()

    return df
