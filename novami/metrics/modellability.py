from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def modi_index(df: pd.DataFrame, desc_col: str, label_col: str, dist_function: str = 'jaccard'):

    df = df.reset_index(drop=True)
    desc_array = np.vstack(df[desc_col].to_numpy())
    dist_array = cdist(XA=desc_array, XB=desc_array, metric=dist_function)

    dist_array[np.eye(len(dist_array), dtype=bool)] = np.inf  # set the diagonal to max to exclude it from search

    # Since this is a distance function, we're looking for a minimum distance

    ddc = defaultdict(lambda: np.zeros((2,)))

    closest_neighbor = np.argmin(dist_array, axis=1)
    for idx, closest_idx in zip(np.arange(len(closest_neighbor)), closest_neighbor):
        idx_class = df.loc[idx, label_col]
        neighbor_class = df.loc[closest_idx, label_col]
        if neighbor_class == idx_class:
            ddc[f'Class_{idx_class}'][0] += 1
        ddc[f'Class_{idx_class}'][1] += 1

    n_classes = df[label_col].nunique()

    modi = np.mean(np.array([values[0]/values[1] for values in ddc.values()]))
    return modi
