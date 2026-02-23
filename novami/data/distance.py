from typing import Optional, Tuple, List, Dict
from joblib import Parallel, delayed
from itertools import chain

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist


def distance_matrix(array_1: np.ndarray, array_2: np.ndarray, metric: str = 'jaccard', n_jobs: int = 1) -> np.ndarray:
    """
    Compute the distance matrix between two arrays using specified metric.

    Parameters
    ----------
    array_1 : numpy.ndarray
        First input array.
    array_2 : numpy.ndarray
        Second input array.
    metric : str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.

    Returns
    -------
    numpy.ndarray
        Distance matrix of shape (array_1.shape[0], array_2.shape[0]).
        Each element (i, j) represents the distance between array_1[i] and array_2[j]
        according to the specified metric.
    """

    if not isinstance(array_1, np.ndarray):
        raise TypeError(f'Expected array_1 to be np.ndarray, got {type(array_1)} instead.')

    if not isinstance(array_2, np.ndarray):
        raise TypeError(f'Expected array_2 to be np.ndarray, got {type(array_2)} instead.')

    if metric not in ['braycurtis', 'canberra', 'chebychev', 'cityblock', 'correlation', 'cosine', 'dice',
                      'euclidean', 'hamming', 'minkowski', 'pnorm', 'jaccard', 'jensenshannon', 'kulczynski1',
                      'mahalanobis', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                      'sqeuclidean', 'sqeuclid', 'yule']:

        raise ValueError(f'Metric {metric} is not supported.')

    if not isinstance(n_jobs, int):
        raise TypeError(f'Expected n_jobs to be int, got {type(n_jobs)} instead.')

    if n_jobs < 1:
        raise ValueError(f'Expected n_jobs to be at least 1, got {n_jobs} instead.')

    def distance_chunk(sub_array_1: np.ndarray, array_2: np.ndarray, metric: str, idx: int) \
            -> Tuple[int, np.ndarray]:

        chunk_distance = cdist(XA=sub_array_1, XB=array_2, metric=metric)
        return idx, chunk_distance

    chunks = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(distance_chunk)(sub_array_1, array_2, metric, idx)
                        for idx, sub_array_1 in enumerate(np.array_split(array_1, n_jobs))
    )

    sorted_chunks = sorted(chunks, key=lambda x: x[0])
    result = np.vstack([chunk for _, chunk in sorted_chunks])

    return result


def k_smallest_rows(array: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k smallest values across each row.
    """

    # Not enough columns, just return the array
    if array.shape[1] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=np.inf)

    col_idx = array.argpartition(k, axis=1)[:, :k]
    row_idx = np.arange(array.shape[0])[:, None]
    return array[row_idx, col_idx]


def k_smallest_columns(array: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k smallest values across each column.
    """

    # Not enough rows, just return the array
    if array.shape[0] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=np.inf)

    row_idx = array.argpartition(k, axis=0)[:k, :]
    col_idx = np.arange(array.shape[1])[None, :]
    return array[row_idx, col_idx]


def k_largest_rows(array: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k largest values across each row.
    """
    if array.shape[1] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=-np.inf)

    col_idx = array.argpartition(-k, axis=1)[:, -k:]
    row_idx = np.arange(array.shape[0])[:, None]
    return array[row_idx, col_idx]


def k_largest_columns(array: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k largest values across each column.
    """
    if array.shape[0] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=-np.inf)

    row_idx = array.argpartition(-k, axis=0)[-k:, :]
    col_idx = np.arange(array.shape[1])[None, :]
    return array[row_idx, col_idx]


def k_neighbors_distance(query_array: np.ndarray, ref_array: np.ndarray = None, metric: str = 'jaccard', n_jobs: int = 1,
                         nearest_k: Optional[List[int]] = None, furthest_k: Optional[List[int]] = None) -> pl.DataFrame:
    """
    Calculate distance statistics between points in query and reference arrays. For each entry in query array,
    calculates the minimum, mean, and maximum distance to all entries in reference array. If it is not passed,
    a self-comparison is made (where distances to self are masked and not included).

    Additionally, computes the average distance to the k nearest and k furthest neighbors,
    where k values are specified in nearest_k and furthest_k parameters.

    Parameters
    ----------
    query_array : np.ndarray
        Query array, typically the test set.
    ref_array : np.ndarray, optional
        Reference array, typically the training set. If not passed, self-comparison.
    metric : str, optional
        Distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is 1.
    nearest_k : Optional[List[int]] or int, optional
        List of k values for which to calculate average distance to the k nearest neighbors.
        Can be a single integer or a list of integers. Default is None, which is equivalent to [1].
    furthest_k : Optional[List[int]] or int, optional
        List of k values for which to calculate average distance to the k furthest neighbors.
        Can be a single integer or a list of integers. Default is None, which is equivalent to [1].

    Returns
    -------
    pl.DataFrame
        A DataFrame containing distance statistics for each entry in query array:
        - 'Min': Minimum distance (equivalent to '1 Nearest')
        - 'Mean': Average distance to all entries in reference array
        - 'Max': Maximum distance (equivalent to '1 Furthest')
        - '{k} Nearest': Average distance to k nearest neighbors for each k in nearest_k
        - '{k} Furthest': Average distance to k furthest neighbors for each k in furthest_k
    """
    self_comparison = False

    if not isinstance(query_array, np.ndarray):
        raise TypeError(f'Expected query array to be np.ndarray, got {type(query_array)} instead.')

    # Self-comparison
    if ref_array is not None:
        if not isinstance(ref_array, np.ndarray):
            raise TypeError(f'Expected reference array to be np.ndarray, got {type(ref_array)} instead.')
    else:
        ref_array = query_array # maybe deepcopy just to be safe?
        self_comparison = True

    if metric not in ['braycurtis', 'canberra', 'chebychev', 'cityblock', 'correlation', 'cosine', 'dice',
                      'euclidean', 'hamming', 'minkowski', 'pnorm', 'jaccard', 'jensenshannon', 'kulczynski1',
                      'mahalanobis', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                      'sqeuclidean', 'sqeuclid', 'yule']:
        raise ValueError(f'Metric {metric} is not supported.')

    if not isinstance(n_jobs, int):
        raise TypeError(f'Expected n_jobs to be int, got {type(n_jobs)} instead.')

    if n_jobs < 1:
        raise ValueError(f'Expected n_jobs to be at least 1, got {n_jobs} instead.')

    if not (isinstance(nearest_k, (int, list)) or nearest_k is None):
        raise TypeError(f'Expected nearest_k to be int or list, got {type(nearest_k)} instead.')
    if not (isinstance(furthest_k, (int, list)) or furthest_k is None):
        raise TypeError(f'Expected furthest_k to be int or list, got {type(furthest_k)} instead.')

    # QoL
    if isinstance(nearest_k, int):
        nearest_k = [nearest_k]
    if isinstance(furthest_k, int):
        furthest_k = [furthest_k]

    # Only find the distance to the most and least similar entry; make sure the nearest/furthest entry is always there
    if nearest_k is None:
        nearest_k = [1]
    else:
        nearest_k = sorted(set(nearest_k + [1]))

    if furthest_k is None:
        furthest_k = [1]
    else:
        furthest_k = sorted(set(furthest_k + [1]))

    # We only need to keep track of these entries for final calculations
    max_n_k = np.max(nearest_k)
    max_f_k = np.max(furthest_k)

    # array.shape[0] % n_jobs arrays of size array.shape[0]//n_jobs + 1
    # and the rest of size array.shape[0] // n_jobs

    splits = np.array_split(ref_array, n_jobs)

    # Only relevant when one array is passed
    k_indices = {0: 0}
    for k, size in enumerate([split.shape[0] for split in splits][:-1], 1):
        k_indices[k] = k_indices[k-1] + size


    def neighbor_chunk(query_array: np.ndarray, sub_ref_array: np.ndarray , metric: str, idx: int, k_indices: Dict,
                       self_comparison: bool, max_n_k: int = 1, max_f_k: int = 1):

        chunk_distance = cdist(XA=sub_ref_array, XB=query_array, metric=metric)

        N = chunk_distance.shape[0]
        M = chunk_distance.shape[1]
        k = k_indices.get(idx)

        if self_comparison:
            chunk_distance[np.eye(N=N, M=M, k=k, dtype=bool)] = np.nan

        # if self_comparison only has 1 compound in a group
        if N > 1:
            chunk_means, chunk_size = np.nanmean(chunk_distance, axis=0), N
        else:
            chunk_means, chunk_size = chunk_distance, 1

        # Find the k nearest/farthest neighbors for given entry
        nearest_distance = k_smallest_columns(chunk_distance, k=max_n_k)
        furthest_distance = k_largest_columns(chunk_distance, k=max_f_k)

        return idx, chunk_means, chunk_size, nearest_distance, furthest_distance

    chunks = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(neighbor_chunk)(
            query_array=query_array,
            sub_ref_array=sub_ref_array,
            metric=metric,
            idx=idx,
            k_indices=k_indices,
            self_comparison=self_comparison,
            max_n_k=max_n_k,
            max_f_k=max_f_k
        )
        for idx, sub_ref_array in enumerate(splits)
    )

    # A single chunk, no need to concatenate anything
    if n_jobs == 1:
        mean_distances = chunks[0][1]
        nearest_array = chunks[0][3]
        furthest_array = chunks[0][4]

    else:
        # Sort by index to ensure the same assignment
        sorted_chunks = sorted(chunks, key=lambda x: x[0])

        # Take weighted average of chunk-means to get the global mean
        means_array = np.vstack([chunk[1] for chunk in sorted_chunks])
        means_sizes = np.array([chunk[2] for chunk in sorted_chunks])
        mean_distances = np.average(means_array, axis=0, weights=means_sizes)

        # Concatenate the results from each chunk
        nearest_array = np.vstack([chunk[3] for chunk in sorted_chunks])
        furthest_array = np.vstack([chunk[4] for chunk in sorted_chunks])

    # Aggregate all the results, find the averaged k-nearest and k-furthest values
    df = (pl.DataFrame({'Mean': mean_distances})
            .with_columns([
                pl.Series(f'{k} Nearest', k_smallest_columns(nearest_array, k=k).mean(axis=0))
                    for k in nearest_k
            ])
            .with_columns([
                pl.Series(f'{k} Furthest', k_largest_columns(furthest_array, k=k).mean(axis=0))
                    for k in furthest_k
            ])
            .rename({'1 Nearest': 'Min', '1 Furthest': 'Max'})
    )

    return df


def group_k_neighbors_distance(df: pl.DataFrame, features_col: str, group_col: str, metric: str = 'jaccard',
                               n_jobs: int = 1):
    """
    Calculate intra- and intergroup distance distribution. The group_col should contain either
    integers or lists of integers.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        The name of the column with features to use for distance calculation
    group_col: str
        The name of the column with group assignments for comparison.
    metric : str, optional
        Distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is 1.

    Returns
    -------
    pl.DataFrame
    """

    req_cols = [features_col, group_col]
    if missing_cols := [col for col in req_cols if col not in df.columns]:
        raise KeyError(f"Missing required columns {missing_cols}")

    # Assume each entry belongs to a single group
    if isinstance(df[group_col].dtype, (pl.Int32, pl.Int64, pl.String)):
        unique_groups = set(df[group_col].to_list())
        print(f"Group column treated a Single-assignment")
        group_type = 'Single'

    # Assume each entry *might* belong to multiple groups
    elif isinstance(df[group_col].dtype, pl.List):
        unique_groups = set(chain.from_iterable(df[group_col].to_list()))
        print(f"Group column treated a Multiple-assignment")
        group_type = 'Multiple'

    else:
        raise TypeError(f"Expected the dtype of group_col to be either Int32, Int64, or String for single group"
                        f"assignment, and pl.List for multi-group assignments. Got {df[group_col].dtype} "
                        f"instead")

    results = []

    for group in unique_groups:
        if group_type == 'Single':
            query_array = np.vstack(df.filter(pl.col(group_col) == group)[features_col].to_numpy())
            ref_array = np.vstack(df.filter(pl.col(group_col) != group)[features_col].to_numpy())
        else:
            query_array = np.vstack(df.filter(pl.col(group_col).list.contains(group))[features_col].to_numpy())
            ref_array = np.vstack(df.filter(~pl.col(group_col).list.contains(group))[features_col].to_numpy())

        # if ref_array is None, calculates distance to self :)
        intra_knd = k_neighbors_distance(
            query_array=query_array,
            ref_array=None,
            metric=metric,
            n_jobs=n_jobs,
        )

        intra_df = pl.DataFrame({
            "Scope": "Intra",
            "Group": group,
            "Aggregation": intra_knd.columns,
            "Quantiles": [np.quantile(intra_knd[col].to_numpy(), np.linspace(0, 1, 11))
                          for col in intra_knd.columns],
        })

        inter_knd = k_neighbors_distance(
            query_array=query_array,
            ref_array=ref_array,
            metric=metric,
            n_jobs=n_jobs,
        )

        inter_df = pl.DataFrame({
            "Scope": "Inter",
            "Group": group,
            "Aggregation": inter_knd.columns,
            "Quantiles": [np.quantile(inter_knd[col].to_numpy(), np.linspace(0, 1, 11))
                          for col in inter_knd.columns],
        })

        results.append(pl.concat([intra_df, inter_df]))

    if not results:
        print("No results")
        return None

    return pl.concat(results)


def list_similarity(string: str, target_strings: list, method: str = 'fuzzy', threshold: float = 0.75,
                    num_matches: int = 3):
    """
    Find the 3 most similar strings in passed target_strings using either fuzzy matching or Levenshtein distance.
    Parameters
    ----------
    string : str
        The input string to compare against the internal database.
    target_strings : list
        The list of strings to search for similarities
    method : str, optional
        The similarity method to use, either 'fuzzy' or 'levenshtein'. Default is 'fuzzy'.
    threshold : float, optional
        The minimum similarity threshold for considering a match. Default is 0.75
    num_matches: int, optional
        Number of top matches to return

    Returns
    -------
    list[tuple]
        A list of up to 3 tuples, each containing a matching string and its similarity score,
        sorted by similarity in descending order.
    """
    try:
        import Levenshtein
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("Function < list_similarity > requires < Levenshtein > and < rapidfuzz > libraries."
                          "Please install them using < pip install Levenshtein rapidfuzz >")

    similarities = []

    for item in target_strings:
        if method == 'levenshtein':
            sim = np.round(1 - Levenshtein.distance(string, item) / max(len(string), len(item)), 3)
        elif method == 'fuzzy':
            sim = np.round(fuzz.ratio(string, item) / 100.0, 3)
        else:
            raise ValueError(f"Available options for method are: levenshtein, fuzzy")
        if sim >= threshold:
            similarities.append((item, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:num_matches]


def dict_similarity(string: str, target_mapping: dict, method: str = 'fuzzy', threshold: float = 0.75,
                    num_matches: int = 3):
    """
    Find the most similar strings in keys of passed mapping using either
    fuzzy matching or Levenshtein distance.

    Parameters
    ----------
    string : str
        The input string to compare against the internal database.
    target_mapping : dict
        The mapping from trade_name to ingredients (dict)
    method : str, optional
        The similarity method to use, either 'fuzzy' or 'levenshtein'. Default is 'fuzzy'.
    threshold : float, optional
        The minimum similarity threshold for considering a match. Default is 0.75
    num_matches: int, optional
        Number of top matches to return

    Returns
    -------
    list[tuple]
        A list of up to num_matches tuples, each containing a matching string and its similarity score,
        sorted by similarity in descending order.
    """
    try:
        import Levenshtein
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("Function < dict_similarity > requires < Levenshtein > and < rapidfuzz > libraries."
                          "Please install them using < pip install Levenshtein rapidfuzz >")

    similarities = []

    for key, value in target_mapping.items():  # trade_name : mixture
        if method == 'levenshtein':
            sim = np.round(1 - Levenshtein.distance(string, key) / max(len(string), len(key)), 3)
        elif method == 'fuzzy':
            sim = np.round(fuzz.ratio(string, key) / 100.0, 3)
        else:
            raise ValueError(f"Available options for method are: levenshtein, fuzzy")

        if sim >= threshold:
            similarities.append((key, value, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)

    return similarities[:num_matches]