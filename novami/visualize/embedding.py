import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def tsne_plot(df: pd.DataFrame, fp_col: str, hue_col: str = None, perplexity: int = 30, save_path: str = None, kwargs=None):

    if kwargs is None:
        kwargs = {}

    df_ = df.copy()

    sns.set_context('notebook')
    sns.set_style('white')

    pca = PCA(n_components=64)
    tsne = TSNE(n_components=2, perplexity=perplexity)

    pca_emb = pca.fit_transform(np.vstack(df_[fp_col].to_numpy()))
    tsne_emb = tsne.fit_transform(pca_emb)
    df_['t-SNE Component 0'] = tsne_emb[:, 0]
    df_['t-SNE Component 1'] = tsne_emb[:, 1]

    if hue_col is not None:
        g = sns.relplot(df_, x='t-SNE Component 0', y='t-SNE Component 1', hue=hue_col, **kwargs)
    else:
        g = sns.relplot(df_, x='t-SNE Component 0', y='t-SNE Component 1', **kwargs)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def umap_plot(df: pd.DataFrame, fp_col: str, hue_col: str = None, umap_kwargs: dict = None,
              save_path: str = None, plot_kwargs: dict = None):
    """
    Prepare UMAP plot of passed dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas Dataframe object.
    fp_col: str
        Name of the column with fingerprints.
    hue_col: str
        Name of the column to be used for coloring the plot.
    umap_kwargs: dict
        Dictionary holding parameters passed to the umap function.
    save_path: str
        If not None, path for saving the plot
    plot_kwargs: dict
        Additional parameters passed to seaborn relplot function.
    """
    random_state = np.random.RandomState(0)

    def_umap_kwargs = {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'metric': 'jaccard'
    }

    def_plot_kwargs = {
        'alpha': 0.8
    }

    if umap_kwargs is not None:
        def_umap_kwargs.update(**umap_kwargs)

    if plot_kwargs is not None:
        def_plot_kwargs.update(**plot_kwargs)

    df_ = df.copy()

    sns.set_context('notebook')
    sns.set_style('white')

    umap_ = umap.UMAP(random_state=random_state, **def_umap_kwargs)

    umap_emb = umap_.fit_transform(X=np.vstack(df_[fp_col].to_numpy()))

    df_['UMAP Component 0'] = umap_emb[:, 0]
    df_['UMAP Component 1'] = umap_emb[:, 1]

    if hue_col is not None:
        g = sns.relplot(df_, x='UMAP Component 0', y='UMAP Component 1', hue=hue_col, **def_plot_kwargs)
    else:
        g = sns.relplot(df_, x='UMAP Component 0', y='UMAP Component 1', **def_plot_kwargs)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')