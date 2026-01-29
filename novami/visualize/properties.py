import os
import warnings
from typing import Union, List

import numpy as np
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors


def continuous_properties(dfs: Union[DataFrame, List[DataFrame]], labels: Union[str, List] = '', label_name: str = 'Dataset',
                          smiles_col: str = 'SMILES', save_directory: str = None, remove_outliers: bool = False,
                          kind: str = 'kde', plot_kwargs: dict = None):
    """
    Generate distribution plots for molecular properties based on SMILES strings.

    This function computes a set of continuous molecular properties (e.g., molecular weight, logP)
    from SMILES representations in the input DataFrame(s) and visualizes their distributions using
    either Kernel Density Estimate (KDE) or violin plots. It supports multiple datasets and provides
    options to handle outliers and customize the appearance of plots.

    Parameters
    ----------
    dfs : Union[DataFrame, List[DataFrame]]
        A DataFrame or list of DataFrames where each contains molecular data, including a SMILES column.
        Each DataFrame corresponds to a distinct dataset.

    labels : Union[str, List]
        A string or a list of strings representing the labels for the datasets provided in `dfs`.
        These labels will be used in the plots to differentiate datasets.

    label_name : str, optional
        The name of the column that will hold the dataset labels in the concatenated DataFrame.
        Default is 'Dataset'.

    smiles_col : str, optional
        The name of the column containing SMILES strings. Default is 'SMILES'.

    save_directory : str, optional
        The directory where the generated plots will be saved. Default is './Figures/'.
        If the directory does not exist, it will be created.

    remove_outliers : bool, optional
        If True, entries outside the range of mean ± 3 standard deviations for each property
        will be removed before plotting to handle extreme outliers. Default is False.

    kind : str, optional
        The type of plot to generate. Either 'kde' for Kernel Density Estimate plots or 'violin' for
        violin plots. Default is 'kde'.

    plot_kwargs : dict, optional
        Additional keyword arguments for customizing the plots (e.g., `linewidths`, `bw_adjust` for KDE,
        or violin-specific settings). These will be passed to the respective seaborn plot functions.
    """

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    assert kind in ['kde', 'violin']

    sns.set_style('white')
    sns.set_context('notebook', font_scale=1.0)

    property_mapping = {
        'MolWt': Descriptors.MolWt,
        'MolLogP': Descriptors.MolLogP,
        'MolMR': Descriptors.MolMR,
        'FractionCSP3': Descriptors.FractionCSP3,
        'TPSA': Descriptors.TPSA,
        'qed': Descriptors.qed
    }

    property_units = {
        'MolWt': 'MolWt (g/mol)',
        'MolLogP': 'MolLogP',
        'MolMR': 'MolMR (cm³/mol)',
        'FractionCSP3': 'Fraction Csp³',
        'TPSA': 'TPSA (Å²)',
        'qed': 'QED'
    }

    properties = list(property_mapping.keys())

    kde_kwargs = {'fill': True, 'alpha': 0.15, 'linewidths': 1.75, 'bw_adjust': 0.7}
    violin_kwargs = {'inner': 'box'}

    if plot_kwargs is not None:
        if kind == 'kde':
            kde_kwargs.update(plot_kwargs)
        elif kind == 'violin':
            violin_kwargs.update(plot_kwargs)

    if isinstance(dfs, DataFrame):
        dfs = [dfs]
    if isinstance(labels, str):
        labels = [labels]

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)

    dfs_ = []

    for df, label in zip(dfs, labels):
        df_ = df.copy()
        df_.loc[:, label_name] = label
        dfs_.append(df_)

    concat_df = pd.concat([df_[[smiles_col, label_name]] for df_ in dfs_], axis=0, ignore_index=True)
    concat_df.loc[:, 'RDKit_mol'] = concat_df[smiles_col].apply(Chem.MolFromSmiles)

    for prop in properties:

        concat_df[prop] = concat_df['RDKit_mol'].apply(
            lambda mol: property_mapping[prop](mol) if mol is not None else np.nan)

        num_entries = len(concat_df)
        concat_df = concat_df.dropna(subset=prop)

        dropped = num_entries - len(concat_df)

        if dropped > 0:
            print(f'Dropped < {dropped} > entries ( {np.round(dropped / num_entries * 100, 3)}% ) '
                  f'during calculation of {prop}')

    if remove_outliers:
        for prop in properties:
            mean = concat_df[prop].mean()
            std = concat_df[prop].std()
            concat_df = concat_df[(concat_df[prop] >= (mean - 3 * std)) & (concat_df[prop] <= (mean + 3 * std))]

    melted_df = pd.melt(concat_df, id_vars=[label_name], value_vars=properties,
                        var_name='Property', value_name='Value')

    melted_df['Property'].replace([np.inf, -np.inf], np.nan, inplace=True)

    g = sns.FacetGrid(melted_df, col='Property', col_wrap=3, sharex=False, sharey=False,
                      hue=label_name, palette='tab10')

    for ax, prop in zip(g.axes.flat, properties):
        prop_df = melted_df[melted_df['Property'] == prop]
        if kind == 'kde':
            sns.kdeplot(prop_df, x='Value', ax=ax, hue=label_name, legend=False, common_norm=False, **kde_kwargs)
        elif kind == 'violin':
            sns.violinplot(prop_df, x=label_name, y='Value', ax=ax, legend=False, common_norm=False, **violin_kwargs)
            ax.set(xticklabels=[])

        ax.set_title(property_units[prop])

    if kind == 'kde':
        g.set_axis_labels(x_var='', y_var='Density')
    elif kind == 'violin':
        g.set_axis_labels(x_var='', y_var='')

    if plot_kwargs is not None and 'palette' in plot_kwargs.keys():
        colors = plot_kwargs['palette']
        handles = []
        for label in labels:
            handles.append(patches.Patch(color=colors[label], label=label, fill=False, linewidth=2))
    else:
        colors = sns.color_palette('tab10', n_colors=len(labels))
        handles = [patches.Patch(color=colors[i], label=labels[i], fill=False, linewidth=2) for i in range(len(labels))]

    g.add_legend(handles=handles, labels=labels, title=label_name, loc='upper right', frameon=True, edgecolor='grey')

    if labels == ['']:
        g._legend.remove()

    if save_directory is not None:
        save_path = os.path.join(save_directory, f'continuous_properties_{kind}.png')
        g.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def discrete_properties(dfs: Union[pd.DataFrame, List[pd.DataFrame]], labels: Union[str, List[str]] = '',
                        label_name: str = 'Dataset', smiles_col: str = 'SMILES', save_directory: str = None,
                        remove_outliers: bool = False, hist_kwargs: dict = None):
    """
    Generate Histogram plots for specified molecular properties from a list of DataFrames.

    This function calculates selected molecular properties from provided SMILES representations,
    concatenates the results into a single DataFrame, and generates Histogram plots to visualize
    the distribution of these properties. Each property is represented in a separate subplot
    within a FacetGrid, with options to filter out outliers based on the standard deviation.

    Parameters
    ----------
    dfs : Union[pd.DataFrame, List[pd.DataFrame]]
        A DataFrame or a list of DataFrames containing molecular data with SMILES strings.

    labels : Union[str, List[str]]
        A label or a list of labels corresponding to each DataFrame in `dfs`.
        Used for distinguishing different datasets in the plots.

    label_name : str, optional
        The name of the column to be created that will hold the labels for each dataset.
        Default is 'Dataset'.

    smiles_col : str, optional
        The name of the column containing SMILES representations of the molecules.
        Default is 'SMILES'.

    save_directory : str, optional
        The directory where the generated plots will be saved. Default is './Figures/'.

    remove_outliers : bool, optional
        If True, entries that fall outside the range of mean ± 3 standard deviations will be removed
        from the DataFrame before plotting. Default is False.

    hist_kwargs : dict, optional
        Additional keyword arguments to customize the histogram plot (e.g., `bins`, `alpha`, etc.).
        Includes default settings for histogram appearance, such as `alpha` for transparency,
        `stat` for normalizing counts, and `discrete` for discrete histograms.
    """

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    sns.set_style('white')
    sns.set_context('notebook', font_scale=1.0)

    property_mapping = {
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHeteroatoms': Descriptors.NumHeteroatoms,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAliphaticRings': Descriptors.NumAliphaticRings,
        'NumAromaticRings': Descriptors.NumAromaticRings
    }

    property_units = {
        'NumHAcceptors': 'HBA',
        'NumHDonors': 'HBD',
        'NumHeteroatoms': 'Heteroatoms',
        'NumRotatableBonds': 'Rotatable Bonds',
        'NumAliphaticRings': 'Aliphatic Rings',
        'NumAromaticRings': 'Aromatic Rings'
    }

    properties = list(property_mapping.keys())

    hist_kws = {
        'shrink': 0.8,
        'alpha': 1
    }

    if hist_kwargs is not None:
        hist_kws.update(hist_kwargs)

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    if isinstance(labels, str):
        labels = [labels]

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)

    for df, label in zip(dfs, labels):
        df[label_name] = label

    concat_df = pd.concat([df[[smiles_col, label_name]] for df in dfs], axis=0, ignore_index=True)
    concat_df['RDKit_mol'] = concat_df[smiles_col].apply(Chem.MolFromSmiles)

    for prop in properties:
        concat_df[prop] = concat_df['RDKit_mol'].apply(
            lambda mol: property_mapping[prop](mol) if mol is not None else np.nan)

        num_entries = len(concat_df)
        concat_df = concat_df.dropna(subset=[prop])

        dropped = num_entries - len(concat_df)

        if dropped > 0:
            print(
                f'Dropped < {dropped} > entries ( {np.round(dropped / num_entries * 100, 3)}% ) during calculation of {prop}')

    if remove_outliers:
        for prop in properties:
            mean = concat_df[prop].mean()
            std = concat_df[prop].std()
            concat_df = concat_df[(concat_df[prop] >= (mean - 3 * std)) & (concat_df[prop] <= (mean + 3 * std))]

    melted_df = concat_df.melt(id_vars=[label_name], value_vars=properties,
                               var_name='Property', value_name='Value')

    melted_df['Value'].replace([np.inf, -np.inf], np.nan, inplace=True)

    g = sns.FacetGrid(melted_df, col='Property', col_wrap=3, sharex=False, sharey=False,
                      hue=label_name, palette='tab10')

    for ax, prop in zip(g.axes.flat, properties):
        prop_df = melted_df[melted_df['Property'] == prop]
        sns.histplot(prop_df, x='Value', ax=ax, discrete=True, hue=label_name, legend=False, multiple='dodge',
                     stat='percent', common_norm=False, **hist_kws)
        ax.set_title(property_units[prop])
        ax.set_xlim(left=-1)

    if hist_kwargs is not None and 'palette' in hist_kwargs.keys():
        colors = hist_kwargs['palette']
        handles = []
        for label in labels:
            handles.append(patches.Patch(color=colors[label], label=label, fill=False, linewidth=2))
    else:
        colors = sns.color_palette('tab10', n_colors=len(labels))
        handles = [patches.Patch(color=colors[i], label=labels[i], fill=False, linewidth=2) for i in range(len(labels))]

    g.add_legend(handles=handles, labels=labels, title=label_name, loc='upper right', frameon=True, edgecolor='grey')

    if labels == ['']:
        g._legend.remove()

    g.set_axis_labels(x_var='', y_var='Percent (%)')

    if save_directory is not None:
        save_path = os.path.join(save_directory, 'discreet_properties.png')
        g.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()