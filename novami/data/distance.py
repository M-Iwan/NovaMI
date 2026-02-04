from typing import Optional, Tuple, List
from joblib import Parallel, delayed

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

    row_idx = array.argpartition(k, axis=0)[:k, :]
    col_idx = np.arange(array.shape[1])[None, :]
    return array[row_idx, col_idx]


def k_largest_rows(array: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k largest values across each row.
    """
    if array.shape[1] <= k:
        return array

    col_idx = array.argpartition(-k, axis=1)[:, -k:]
    row_idx = np.arange(array.shape[0])[:, None]
    return array[row_idx, col_idx]


def k_largest_columns(array: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k largest values across each column.
    """
    if array.shape[0] <= k:
        return array

    row_idx = array.argpartition(-k, axis=0)[-k:, :]
    col_idx = np.arange(array.shape[1])[None, :]
    return array[row_idx, col_idx]


def k_neighbors_distance(array_1: np.ndarray, array_2: np.ndarray, metric: str = 'jaccard', n_jobs: int = 1,
                         nearest_k: Optional[List[int]] = None, furthest_k: Optional[List[int]] = None) -> pl.DataFrame:
    """
    Calculate distance statistics between points in array_2 and array_1.

    For each entry in array_2, calculates the minimum, mean, and maximum distance to all entries in array_1.
    Additionally, computes the average distance to the k nearest and k furthest neighbors,
    where k values are specified in nearest_k and furthest_k parameters.

    Parameters
    ----------
    array_1 : numpy.ndarray
        Reference array, typically the training set.
    array_2 : numpy.ndarray
        Query array, typically the test set.
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
        A DataFrame containing distance statistics for each entry in array_2:
        - 'Min': Minimum distance (equivalent to '1_nearest')
        - 'Mean': Average distance to all entries in array_1
        - 'Max': Maximum distance (equivalent to '1_furthest')
        - '{k} Nearest': Average distance to k nearest neighbors for each k in nearest_k
        - '{k} Furthest': Average distance to k furthest neighbors for each k in furthest_k
    """

    if not isinstance(array_1, np.ndarray):
        raise TypeError(f'Expected array_1 to be np.ndarray, got {type(array_1)} instead.')

    if not isinstance(array_2, np.ndarray):
        raise TypeError(f'Expected array_1 to be np.ndarray, got {type(array_2)} instead.')

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

    def neighbor_chunk(sub_array_1: np.ndarray, array_2: np.ndarray, metric: str, idx: int,
                       max_n_k: int = 1, max_f_k: int = 1):

        chunk_distance = cdist(XA=sub_array_1, XB=array_2, metric=metric)

        chunk_means, chunk_size = chunk_distance.mean(axis=0), chunk_distance.shape[0]

        # Find the k nearest/farthest neighbors for given entry
        nearest_distance = k_smallest_columns(chunk_distance, k=max_n_k)
        furthest_distance = k_largest_columns(chunk_distance, k=max_f_k)

        return idx, chunk_means, chunk_size, nearest_distance, furthest_distance

    chunks = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(neighbor_chunk)(sub_array_1, array_2, metric, idx, max_n_k, max_f_k)
        for idx, sub_array_1 in enumerate(np.array_split(array_1, n_jobs))
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