from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

def good_plot(df: pl.DataFrame, threshold_col: str, performance_col: str, save_path: Optional[str] = None,
              plot_kwargs: Optional[dict] = None):

    # set_font()
    sns.set_style('whitegrid')
    sns.set_context('paper')

    dparams = {}
    if plot_kwargs is not None:
        dparams.update(plot_kwargs)

    g = sns.relplot(df, x=threshold_col, y=performance_col, kind='line', **dparams)

    g.set_xlabels('Distance threshold')
    g.set_ylabels('Model performance')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()