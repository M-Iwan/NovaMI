
def morgan_intra_similarities(dfs: Union[DataFrame, List[DataFrame]], labels: Union[str, List],
                            label_name: str = 'Dataset', smiles_col: str = 'SMILES',
                            save_directory: str = None, radius: int = 2, nbits: int = 2048,
                            limits: list = None):
    """
    Calculate and plot the cumulative distribution functions (CDFs) of molecular
    similarity based on Morgan fingerprints from multiple datasets.

    The function computes pairwise Tanimoto similarity for the molecules in the provided datasets,
    aggregates the similarities using minimum, mean, and maximum statistics,
    interpolates the CDFs to a standard size, and visualizes the results.

    Parameters
    ----------
    dfs : Union[DataFrame, List[DataFrame]]
        A DataFrame or a list of DataFrames containing molecular data. Each DataFrame
        must include at least the columns specified by `smiles_col` and a computed
        Morgan fingerprint.

    labels : Union[str, List]
        A string or list of strings that serve as labels for the datasets.
        Each label corresponds to a DataFrame in `dfs`.

    label_name : str, optional
        The name of the column that will represent the dataset labels in the final DataFrame.
        Defaults to 'Dataset'.

    smiles_col : str, optional
        The name of the column containing the SMILES representations of the molecules.
        Defaults to 'SMILES'.

    save_directory : str, optional
        The directory path where the resulting plots will be saved. Defaults to './Figures/'.

    radius : int, optional
        The radius parameter for the Morgan fingerprint calculation, determining the
        size of the neighborhood around each atom to consider. Defaults to 2.

    nbits : int, optional
        The number of bits to use in the Morgan fingerprints, impacting the
        dimensionality of the representation. Defaults to 2048.

    limits: list, optional
        Manually specify the xlim for each subplot. The expected format is
        List[List[float, float]] with 3 nested lists (for min, mean, max)

    Notes
    -----
    Using 'jaccard' distance from scipy returns Jaccard Distance, which is a dissimilarity measure.
    Therefore, identical vectors will return '0', while completely different vectors (on basis of
    dissimilar, non-zero elements) will return '1'.

    Taking 1 - Jaccard Distance gives similarity measure that is equivalent to Tanimoto Similarity, and also
    more logical (bigger values -> more similar)
    """

    aggrs = ['Minimum Similarity', 'Mean Similarity', 'Maximum Similarity']
    if limits is None:
        limits = [[], [], []]

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    sns.set_style('whitegrid')
    sns.set_context('notebook', font_scale=1.0)

    if isinstance(dfs, DataFrame):
        dfs = [dfs]
    if isinstance(labels, str):
        labels = [labels]

    dfs_ = []
    for df, label in zip(dfs, labels):
        smis = list(df[smiles_col].unique())
        morgan_array = np.vstack(smiles_2_morgan(smis, radius=radius, nbits=nbits))
        morgan_sim = 1 - cdist(XA=morgan_array, XB=morgan_array, metric='jaccard')  # equivalent to Tanimoto Similarity
        del morgan_array
        morgan_sim = morgan_sim[~np.eye(len(morgan_sim), dtype=bool)].reshape(morgan_sim.shape[0], -1)  # remove diagonal, as the similarity is always = 1 there

        cdf_min = np.array(sorted(morgan_sim.min(axis=1)))  # maybe to be removed, as not very informative
        cdf_mean = np.array(sorted(morgan_sim.mean(axis=1)))
        cdf_max = np.array(sorted(morgan_sim.max(axis=1)))

        df_ = pd.DataFrame({'Minimum': cdf_min, 'Mean': cdf_mean, 'Maximum': cdf_max})

        df_.loc[:, label_name] = label

        dfs_.append(df_)

    concat_df = pd.concat(dfs_, axis=0, ignore_index=True)

    melted_df = pd.melt(concat_df, id_vars='Dataset', value_vars=['Minimum','Mean', 'Maximum'], var_name='Property',
                        value_name='Cumulative Probability')

    g = sns.FacetGrid(concat_df, col='Dataset', sharex=False, sharey=False, palette='tab10')

    for ax, aggr, lims in zip(g.axes.flat, aggrs, limits):
        aggr_df = melted_df[melted_df['Property'] == aggr]
        sns.ecdfplot(aggr_df, x='Cumulative Probability', hue='Dataset', ax=ax, legend=False)
        ax.set_title(aggr)
        ax.set_ylim(-0.05, 1.05)
        if lims:
            ax.set_xlim(lims[0], lims[1])

    g.set_axis_labels(x_var='Similarity', y_var='Cumulative Probability')

    colors = sns.color_palette('tab10', n_colors=len(aggrs))
    handles = [patches.Patch(color=colors[i], label=aggrs[i], fill=False, linewidth=2)
               for i in range(len(aggrs))]

    g.add_legend(handles=handles, labels=labels, title=label_name, loc='upper right', frameon=True, edgecolor='grey')

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, 'intra_ecdf.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def morgan_intra_similarity(df: pd.DataFrame, smiles_col: str = 'SMILES',
                            save_directory: str = None, radius: int = 2, nbits: int = 2048):
    """
    Calculate the inter-similarity between training and test sets using Morgan fingerprints
    and generate empirical cumulative distribution function (eCDF) plots.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing SMILES strings.

    smiles_col: str, optional
        The column name in the DataFrames that contains the SMILES strings (default is 'SMILES').

    save_directory: str, optional
        Directory to save the eCDF plot. If None, the plot will not be saved (default is None).

    radius: int, optional
        Radius for generating Morgan fingerprints (default is 2).

    nbits: int, optional
        Number of bits for the Morgan fingerprint representation (default is 2048).
    """

    sns.set_style('whitegrid')
    sns.set_context('notebook', font_scale=1.0)

    x_smi = list(df[smiles_col].unique())

    x = np.vstack(smiles_2_morgan(x_smi, radius=radius, nbits=nbits))

    morgan_sim = 1 - cdist(XA=x, XB=x, metric='jaccard')
    morgan_sim = morgan_sim[~np.eye(len(morgan_sim), dtype=bool)].reshape(morgan_sim.shape[0], -1)

    cdf_min = np.array(sorted(morgan_sim.min(axis=0)))
    cdf_mean = np.array(sorted(morgan_sim.mean(axis=0)))
    cdf_max = np.array(sorted(morgan_sim.max(axis=0)))

    sim_df = pd.DataFrame({'Minimum': cdf_min, 'Mean': cdf_mean, 'Maximum': cdf_max})
    melt_df = pd.melt(sim_df, value_vars=['Minimum', 'Mean', 'Maximum'], var_name='Similarity', value_name='Tanimoto Similarity')
    g = sns.ecdfplot(melt_df, x='Tanimoto Similarity', hue='Similarity')
    g.set_xlabel('Tanimoto Similarity')
    g.set_ylabel('Cumulative Probability')

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, 'inter_ecdf.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def morgan_inter_similarity(train: pd.DataFrame, test: pd.DataFrame, smiles_col: str = 'SMILES',
                            save_directory: str = None, radius: int = 2, nbits: int = 2048):
    """
    Calculate the inter-similarity between training and test sets using Morgan fingerprints
    and generate empirical cumulative distribution function (eCDF) plots.

    Parameters
    ----------
    train : pd.DataFrame
        A DataFrame containing the training set with SMILES strings.

    test : pd.DataFrame
        A DataFrame containing the test set with SMILES strings.

    smiles_col : str, optional
        The column name in the DataFrames that contains the SMILES strings (default is 'SMILES').

    save_directory : str, optional
        Directory to save the eCDF plot. If None, the plot will not be saved (default is None).

    radius : int, optional
        Radius for generating Morgan fingerprints (default is 2).

    nbits : int, optional
        Number of bits for the Morgan fingerprint representation (default is 2048).
    """

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    sns.set_style('whitegrid')
    sns.set_context('notebook', font_scale=1.0)

    xa_smi = list(train[smiles_col].unique())
    xb_smi = list(test[smiles_col].unique())

    xa = np.vstack(smiles_2_morgan(xa_smi, radius=radius, nbits=nbits))
    xb = np.vstack(smiles_2_morgan(xb_smi, radius=radius, nbits=nbits))

    morgan_sim = 1 - cdist(XA=xa, XB=xb, metric='jaccard')

    cdf_min = np.array(sorted(morgan_sim.min(axis=0)))
    cdf_mean = np.array(sorted(morgan_sim.mean(axis=0)))
    cdf_max = np.array(sorted(morgan_sim.max(axis=0)))

    sim_df = pd.DataFrame({'Minimum': cdf_min, 'Mean': cdf_mean, 'Maximum': cdf_max})
    melt_df = pd.melt(sim_df, value_vars=['Minimum', 'Mean', 'Maximum'], var_name='Similarity', value_name='Tanimoto Similarity')
    g = sns.ecdfplot(melt_df, x='Tanimoto Similarity', hue='Similarity')
    g.set_xlabel('Tanimoto Similarity')
    g.set_ylabel('Cumulative Probability')

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, 'inter_ecdf.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()