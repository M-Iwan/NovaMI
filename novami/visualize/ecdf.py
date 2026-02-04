from typing import Union, List, Optional

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from novami.visualize.utils import *
from novami.data.distance import k_neighbors_distance


def k_neighbours_ecdf(train_df: pl.DataFrame, test_df: pl.DataFrame, embedding_col: str, metric: str,
                      nearest_k: Optional[List[int]] = None, furthest_k: Optional[List[int]] = None,
                      n_jobs: Optional[int] = 1, save_path: Optional[str] = None, plot_kwargs: Optional[dict] = None):

    set_font()
    sns.set_style('whitegrid')
    sns.set_context('paper')
    mean_col, min_col, max_col = three_point_palette()

    dparams = {}
    if plot_kwargs is not None:
        dparams.update(plot_kwargs)

    # Making the palette and order correct in an absolutely atrocious way...
    palette = {'Mean': mean_col}
    hue_order = []

    if nearest_k is not None:
        min_colours = generate_color_variants(min_col, len(nearest_k), step=0.1)
        for k, colour in zip(sorted(nearest_k), min_colours):
            if k == 1:
                palette['Min'] = colour
                hue_order.append('Min')
            else:
                palette[f'{k} Nearest'] = colour
                hue_order.append(f'{k} Nearest')
    else:
        palette['Min'] = min_col
        hue_order.append('Min')

    hue_order.append('Mean')

    if furthest_k is not None:
        max_colours = generate_color_variants(max_col, len(furthest_k))
        for k, colour in zip(sorted(furthest_k)[::-1], max_colours[::-1]):
            if k == 1:
                palette['Max'] = colour
                hue_order.append('Max')
            else:
                palette[f'{k} Furthest'] = colour
                hue_order.append(f'{k} Furthest')
    else:
        palette['Max'] = max_col
        hue_order.append('Max')

    if embedding_col not in train_df.columns or embedding_col not in test_df.columns:
        raise KeyError(f'{embedding_col} not found in DataFrame.')

    array_1 = train_df[embedding_col].to_numpy()
    array_2 = test_df[embedding_col].to_numpy()

    neighbor_df = k_neighbors_distance(
        array_1=array_1,
        array_2=array_2,
        metric=metric,
        n_jobs=n_jobs,
        nearest_k=nearest_k,
        furthest_k=furthest_k)

    neighbor_df = neighbor_df.unpivot(on=None, variable_name='Aggregation', value_name='Distance')

    g = sns.ecdfplot(data=neighbor_df, x='Distance', hue='Aggregation', palette=palette, hue_order=hue_order)

    g.set_xlabel(f"{metric.capitalize()} Distance")
    g.set_ylabel('Cumulative Probability')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
