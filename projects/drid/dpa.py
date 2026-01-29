"""
Module with functions for Disproportionality Analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
import itertools
from typing import Union, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype


def is_valid_float(value):
    return isinstance(value, float) and not np.isnan(value)


def binarize_reactions(df: pd.DataFrame, pt_idx: List[int], strat_col: Optional[Union[str, List[str]]] = None, smiles_col: str = 'smi_enc',
                       reaction_col: str = 'reac_enc', count_col: str = 'count'):
    """
    Combine reactions based on passed pt_idx to obtain a binary classification.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame with smi_enc : reac_enc counts
    pt_idx: List[int]
        List of reaction encoding indices to merge
    strat_col: Union[None, str, List[str]]
        List of columns to stratify on, e.g. ['sex', 'age', 'weight'].
        The smiles_col and reaction_col will be appended to the passed list.
        Default is None.
    smiles_col: str, optional
        Name of the column with drug/SMILES description. Default is 'smi_enc'
    reaction_col: str, optional
        Name of the column with event description. Default is 'reac_enc'
    count_col: str, optional
        Name of the column with pair counts. Default is 'count'

    Returns
    -------
    df: pd.DataFrame
        Pandas DataFrame with binarized reaction column
    """

    df = df.reset_index(drop=True)
    df.loc[:, reaction_col] = df[reaction_col].apply(lambda reaction: int(reaction in pt_idx))

    if strat_col is None:
        strat = [smiles_col, reaction_col]
    elif isinstance(strat_col, str):
        strat = [strat_col, smiles_col, reaction_col]
    elif isinstance(strat_col, list):
        strat = strat_col + [smiles_col, reaction_col]
    else:
        raise ValueError(f'Expected strat_col to be one of < None, int, list >. Received < {type(strat_col)} > instead')

    df = df.groupby(strat)[count_col].sum().reset_index(name=count_col)
    return df


def dpa_matrix(df: pd.DataFrame, smi_idx: int, pt_idx: Union[int, List[int]],
               smiles_col: str = 'StSMILES_enc', reaction_col: str = 'reactions_enc'):
    """
    Calculate the contingency matrix for a given SMILES index : PT index pair(s).

    Parameters
    ----------
    df: pd.DataFrame
        Dataset to be used. Must include columns corresponding to drug and event in 'exploded' form.
    smi_idx: int
        SMILES index to use.
    pt_idx: Union[int, List[int]]
        PT index or list of indices to use.
    smiles_col: str, optional
        Name of the column with drug/SMILES description.
    reaction_col: str, optional
        Name of the column with event description.

    Returns
    -------
    cont_matrix: dict
        Dictionary with the results.
    """

    smiles_mask = df[smiles_col] == smi_idx

    reaction_mask = df[reaction_col].isin(pt_idx) if isinstance(pt_idx, list) else (df[reaction_col] == pt_idx)

    a = np.sum(smiles_mask & reaction_mask)  # pairs of drug AND event
    b = np.sum(smiles_mask & ~reaction_mask)  # pairs of drug AND NOT event
    c = np.sum(~smiles_mask & reaction_mask)  # paris of NOT drug AND event
    d = np.sum(~smiles_mask & ~reaction_mask)  # pairs of NOT drug AND NOT event

    cont_matrix = {smiles_col: [smi_idx], reaction_col: [pt_idx],
                   "a": a, "b": b, "c": c, "d": d}

    return cont_matrix


def subpopulation_agreement(df: pd.DataFrame, strat_col: str, strat_names: List[str], smiles_col: str = 'SMILES'):
    """
    Calculate the fraction of common SMILES between two datasets.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe in a long format (melted on strat_col).
    strat_col: str
        Column to stratify on. Ex: 'Sex', 'Age'.
    strat_names: str
        Sub-categories of stratification column. Ex. ['Neonate', 'Adult', 'Elderly']
    smiles_col: str
        Column holding SMILES.

    Returns
    -------
    df: pd.DataFrame
        Pandas dataframe with results.
    """
    rows = []
    for pair in itertools.product(strat_names, repeat=2):
        item_1, item_2 = pair
        set_1 = set(df[df[strat_col] == item_1][smiles_col])
        set_2 = set(df[df[strat_col] == item_2][smiles_col])
        total = len(set_1.union(set_2))
        common = set_1.intersection(set_2)
        set_1_unique = set_1.difference(set_2)
        set_2_unique = set_2.difference(set_1)
        frac = np.round((len(common) / total), 3)
        frac_ = f'{frac:.3f}'
        agr = np.round(len(common) / min([len(set_1), len(set_2)]), 3)
        agr_ = f'{agr:.3f}'

        row = pd.DataFrame({f'{strat_col}_1': [item_1], f'{strat_col}_2': [item_2], 'num_total': [total], 'common': [common], 'num_common': [len(common)],
                            f'{strat_col}_1_unique': [set_1_unique], f'num_{strat_col}_1_unique': [len(set_1_unique)],
                            f'{strat_col}_2_unique': [set_2_unique], f'num_{strat_col}_2_unique': [len(set_2_unique)],
                            'frac': [frac], 'frac_str': [f"{frac_}\n{str(len(common))}/{str(total)}"],
                            'agr': [agr], 'agr_str': [f"{agr_}\n{str(len(common))}/{str(total)}"]})
        rows.append(row)
    df = pd.concat(rows).reset_index(drop=True)
    return df


def plot_agreement(df: pd.DataFrame, strat: str, order: List[str], title: str, save_path: str, frac_type: str = 'frac'):
    """
    Plot the heatmap showcasing the agreement between
    """
    df_p = df.pivot(index=f'{strat}_1', columns=f'{strat}_2', values=frac_type)
    df_a = df.pivot(index=f'{strat}_1', columns=f'{strat}_2', values=f'{frac_type}_str')
    df_p = df_p.loc[order, order[::-1]]
    df_a = df_a.loc[order, order[::-1]]

    g = sns.heatmap(df_p, cmap='magma', fmt='s', vmin=0, vmax=1, annot=df_a)
    plt.xlabel(strat.capitalize())
    plt.ylabel(strat.capitalize())
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return g


def multi_subpopulation_agreement(df_1: pd.DataFrame, df_2: pd.DataFrame, strat_col: str, strat_names_1: List[str],
                                  strat_names_2: List[str], smiles_col: str = 'SMILES'):

    rows = []
    for pair in itertools.product(strat_names_1, strat_names_2):
        item_1, item_2 = pair
        set_1 = set(df_1[df_1[strat_col] == item_1][smiles_col])
        set_2 = set(df_2[df_2[strat_col] == item_2][smiles_col])
        total = len(set_1.union(set_2))
        common = set_1.intersection(set_2)
        set_1_unique = set_1.difference(set_2)
        set_2_unique = set_2.difference(set_1)
        frac = np.round((len(common) / total), 3)
        frac_ = f'{frac:.3f}'
        agr = np.round(len(common) / min([len(set_1), len(set_2)]), 3)
        agr_ = f'{agr:.3f}'

        row = pd.DataFrame({f'{strat_col}_1': [item_1], f'{strat_col}_2': [item_2], 'num_total': [total], 'common': [common], 'num_common': [len(common)],
                            f'{strat_col}_1_unique': [set_1_unique], f'num_{strat_col}_1_unique': [len(set_1_unique)],
                            f'{strat_col}_2_unique': [set_2_unique], f'num_{strat_col}_2_unique': [len(set_2_unique)],
                            'frac': [frac], 'frac_str': [f"{frac_}\n{str(len(common))}/{str(total)}"],
                            'agr': [agr], 'agr_str': [f"{agr_}\n{str(len(common))}/{str(total)}"]})
        rows.append(row)

    df = pd.concat(rows).reset_index(drop=True)
    return df


def plot_multi_agreement(df: pd.DataFrame, strat: str, order_1: List[str], order_2: List[str], title: str, save_path: str, frac_type: str = 'frac'):
    """
    Plot the heatmap showcasing the multi agreement
    """
    df_p = df.pivot(index=f'{strat}_1', columns=f'{strat}_2', values=frac_type)
    df_a = df.pivot(index=f'{strat}_1', columns=f'{strat}_2', values=f'{frac_type}_str')
    df_p = df_p.loc[order_1, order_2[::-1]]
    df_a = df_a.loc[order_1, order_2[::-1]]

    g = sns.heatmap(df_p, cmap='magma', fmt='s', vmin=0, vmax=1, annot=df_a)
    plt.xlabel(strat.capitalize())
    plt.ylabel(strat.capitalize())
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return g


"""
Step 6
Prepare the drug-reaction counts stratified by Demo Factors.
"""


def step_6_1(df: pd.DataFrame):
    """
    Prepare pair counts, stratified by sex/age.
    """

    strats = [(['Sex'], 's'), (['Age'], 'a'), (['Weight'], 'w'), (['Sex', 'Age'], 'sa'), (['Sex', 'Weight'], 'sw'),
              (['Age', 'Weight'], 'aw'), (['Sex', 'Age', 'Weight'], 'saw')]

    df = df.explode('reac_enc').explode('smi_enc').reset_index(drop=True)
    df = df.astype({'reac_enc': 'UInt16', 'smi_enc': 'UInt16'})

    dfs = [df.groupby(['smi_enc', 'reac_enc']).size().reset_index(name='count').assign(strat='n')]

    for strat, name in strats:
        sub_df = df.groupby(strat).apply(lambda group: group.groupby(['smi_enc', 'reac_enc']).size()).reset_index(name='count').assign(strat=name)
        dfs.append(sub_df)

    df = pd.concat(dfs).reset_index(drop=True)

    return df


def step_6_2(dfs: pd.DataFrame):
    """
    Finally combine split datasets
    """

    strats = {'s': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    pr_dfs = [dfs[dfs['strat'] == 'n'].groupby(['smi_enc', 'reac_enc'])['count'].sum().reset_index(name='count').assign(strat='n')]

    for name, strat in strats.items():
        sub_df = dfs[dfs['strat'] == name]
        gb = sub_df.groupby(strat).apply(lambda group: group.groupby(['smi_enc', 'reac_enc'])['count'].sum().reset_index(name='count')).reset_index()
        gb = gb.drop(columns=[f'level_{len(name)}']).assign(strat=name)

        pr_dfs.append(gb)

    return pd.concat(pr_dfs, ignore_index=True)


"""
Step 7
"""


def step_7(df: pd.DataFrame, idx_dc: dict, strat_type: str):
    """
    Assign (again) cardiotoxicity labels to reactions.
    """
    idx_abbrev = {}

    abbrev_mapping = {
        'cardio_reduced': 'CRed',
        'cardio': 'Card',
        'cardiovasc': 'CVasc',
        'cardiac arrhythmias': 'CA',
        'myocardial disorders': 'MD',
        'coronary artery disorders': 'CD',
        'heart failures': 'HF',
        'pericardial disorders': 'PD',
        'endocardial disorders': 'ED',
        'cardiac valve disorders': 'CV'
    }

    for key, values in idx_dc.items():
        idx_abbrev[abbrev_mapping.get(key)] = values

    strats = {'s': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    dfs = []

    for key, values in idx_abbrev.items():
        sub_df = df.copy()
        sub_df.loc[:, key] = df['reac_enc'].apply(lambda entry: int(entry in values)).astype('UInt8')

        if strat_type != 'n':
            sub_df = sub_df.groupby(strats.get(strat_type)).apply(lambda group: group.groupby(['smi_enc', key])['count'].sum().reset_index(name=f'{key}_ct'))
            sub_df = sub_df.reset_index().drop(columns=f'level_{len(strat_type)}')
        else:
            sub_df = sub_df.groupby(['smi_enc', key])['count'].sum().reset_index(name=f'{key}_ct')
        dfs.append((sub_df, key, strat_type))

    return dfs


"""
Step 8
"""


def dpa_matrix_count(df: pd.DataFrame, smi_idx: int, pt_idx: Union[int, List[int]], smiles_col: str = 'smi_enc',
                     reaction_col: str = 'reac_enc', count_col: str = 'count'):
    """
    Calculate the contingency matrix for a given SMILES index : PT index pair(s) using pre-counted pairs.
    The paris can be obtained using count_pairs function.

    Parameters
    ----------
    df: pd.DataFrame
        Dataset to be used. Must include columns corresponding to drug and event in 'exploded' form.
    smi_idx: int
        SMILES index to use.
    pt_idx: Union[int, List[int]]
        PT index or list of indices to use.
    smiles_col: str, optional
        Name of the column with drug/SMILES description. Default is 'smi_enc'
    reaction_col: str, optional
        Name of the column with event description. Default is 'reac_enc'
    count_col: str, optional
        Name of the column with pair counts. Default is 'count'

    Returns
    -------
    cont_matrix: dict
        Dictionary holding the results.
    """

    if isinstance(pt_idx, int):
        reaction_mask = df[reaction_col] == pt_idx
    else:
        reaction_mask = df[reaction_col].isin(pt_idx)

    a = df[(df[smiles_col] == smi_idx) & reaction_mask][count_col].sum()  # pairs of drug AND reaction
    b = df[(df[smiles_col] == smi_idx) & ~reaction_mask][count_col].sum()  # pairs of drug AND NOT reaction
    c = df[(df[smiles_col] != smi_idx) & reaction_mask][count_col].sum()  # pairs of NOT drug and reaction
    d = df[(df[smiles_col] != smi_idx) & ~reaction_mask][count_col].sum()  # pairs of NOT drug and NOT reaction

    return {smiles_col: smi_idx, reaction_col: pt_idx,
            'a': a, 'b': b, 'c': c, 'd': d}


def step_8(df: pd.DataFrame, smiles_col: str, reaction_col: str, count_col: str, pt_idx: int, strat_type: str):

    strats = {'s': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    if strat_type == 'n':
        entries = [dpa_matrix_count(df, smi_idx, pt_idx, smiles_col, reaction_col, count_col) for smi_idx in df[smiles_col].unique()]
        ct_df = pd.DataFrame(entries)

    else:
        dfs = []
        strat = strats.get(strat_type)
        for group in df.groupby(strat):
            group_df = pd.DataFrame(group[1])
            entries = [dpa_matrix_count(group_df, smi_idx, pt_idx, smiles_col, reaction_col, count_col) for smi_idx in group_df[smiles_col].unique()]
            df_ = pd.DataFrame(entries).assign(**{strat[i]: group[0][i] for i in range(len(strat))})
            dfs.append(df_)
        ct_df = pd.concat(dfs, ignore_index=True)

    ct_df = ct_df.astype({reaction_col: 'UInt8', 'a': 'UInt32', 'b': 'UInt32', 'c': 'UInt32', 'd': 'UInt32'})
    return ct_df


"""
Step 9
"""


def proportional_reporting_rate(a: Union[int, np.ndarray], b: Union[int, np.ndarray], c: Union[int, np.ndarray],
                                d: Union[int, np.ndarray], sign_level: float = 0.05, decimals: int = 5):
    """
    Calculate the Proportional Reporting Ratio (PRR) from a contingency table. Based on  M. Fusaroli 'pvda' R package.
    Divisions by zero errors are caught automatically and return np.nan.

    Parameters
    ----------
    a: Union[int, np.ndarray]
        Number of drug AND event pairs
    b: Union[int, np.ndarray]
        Number of drug AND NOT event pairs
    c: Union[int, np.ndarray]
        Number of NOT drug AND event pairs
    d: Union[int, np.ndarray]
        Number of NOT drug AND NOT event pairs
    sign_level: float, optional
        Significance level when calculating the CI. Default is 0.05
    decimals: int, optional
        Number of decimals to keep. Default is 5

    Returns
    -------
    prr_value: Union[float, np.ndarray]
        Calculated PRR.
    prr_lower: Union[float, np.ndarray]
        Lower bound of the PRR Confidence Interval.
    prr_upper: Union[float, np.ndarray]
        Upper bound of the PRR Confidence Interval.
    """

    exp_count = ((a + b) * c) / (c + d)
    prr_value = np.round(a / exp_count, decimals)

    s = np.sqrt((1/a) - (1/(a+b)) + (1/c) - (1/(c + d)))
    z_value = norm.ppf(1 - (sign_level / 2))

    prr_lower = np.round(prr_value * np.exp(-z_value * s), decimals)
    prr_upper = np.round(prr_value * np.exp(z_value * s), decimals)

    return prr_value, prr_lower, prr_upper


def reporting_odds_ratio(a: Union[int, np.ndarray], b: Union[int, np.ndarray], c: Union[int, np.ndarray],
                         d: Union[int, np.ndarray], sign_level: float = 0.05, decimals: int = 5):
    """
    Calculate the Reporting Odds Ratio (ROR) from a contingency table. Based on  M. Fusaroli 'pvda' R package.
    Divisions by zero errors are caught automatically and return np.nan.

    Parameters
    ----------
    a: Union[int, np.ndarray]
        Number of drug AND event pairs
    b: Union[int, np.ndarray]
        Number of drug AND NOT event pairs
    c: Union[int, np.ndarray]
        Number of NOT drug AND event pairs
    d: Union[int, np.ndarray]
        Number of NOT drug AND NOT event pairs
    sign_level: float, optional
        Significance level when calculating the CI. Default is 0.05
    decimals: int, optional
        Number of decimals to keep. Default is 5

    Returns
    -------
    ror_value: Union[float, np.ndarray]
        Calculated ROR.
    ror_lower: Union[float, np.ndarray]
        Lower bound of the ROR Confidence Interval.
    ror_upper: Union[float, np.ndarray]
        Upper bound of the ROR Confidence Interval.
    """

    ror_value = np.round((a*d) / (b*c), decimals)

    s = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z_value = norm.ppf(1 - (sign_level / 2))

    ror_lower = np.round(ror_value * np.exp(-z_value * s), decimals)
    ror_upper = np.round(ror_value * np.exp(z_value * s), decimals)

    return ror_value, ror_lower, ror_upper


def information_component(a: Union[int, np.ndarray], b: Union[int, np.ndarray], c: Union[int, np.ndarray], d: Union[int, np.ndarray],
                          sign_level: float = 0.05, shrink: float = 0.5, decimals: int = 5):
    """
    Calculate the Information Component (IC) from a contingency table. Based on  M. Fusaroli 'pvda' R package.

    Parameters
    ----------
    a: Union[int, np.ndarray]
        Number of drug AND event pairs
    b: Union[int, np.ndarray]
        Number of drug AND NOT event pairs
    c: Union[int, np.ndarray]
        Number of NOT drug AND event pairs
    d: Union[int, np.ndarray]
        Number of NOT drug AND NOT event pairs
    sign_level: float, optional
        Significance level when calculating the CI. Default is 0.05
    shrink: float, optional
        Shrinkage factor. Default is 0.5.
    decimals: int, optional
        Number of decimals to keep. Default is 5

    Returns
    -------
    ic_value: Union[float, np.ndarray]
        Calculated Information Content.
    ic_lower: Union[float, np.ndarray]
        Lower bound of the IC Confidence Interval.
    ic_upper: Union[float, np.ndarray]
        Upper bound of the IC Confidence Interval.
    """

    obs_count = a
    exp_count = (a + b) * (a + c) / (a + b + c + d)
    # n_drugs * n_events / n_total
    alpha = obs_count + shrink
    beta = exp_count + shrink

    ic_value = np.round(np.log2(alpha / beta), decimals)
    ic_lower = np.round(np.log2(gamma.ppf(sign_level / 2, a=alpha, scale=1/beta)), decimals)
    ic_upper = np.round(np.log2(gamma.ppf(1 - sign_level / 2, a=alpha, scale=1/beta)), decimals)

    return ic_value, ic_lower, ic_upper


def calculate_dpa_metrics(df, a_col='a', b_col='b', c_col='c', d_col='d', sign_level: float = 0.05, shrink: float = 0.5):

    a = df[a_col].to_numpy()
    b = df[b_col].to_numpy()
    c = df[c_col].to_numpy()
    d = df[d_col].to_numpy()

    prr_values, prr_lower, prr_upper = proportional_reporting_rate(a=a, b=b, c=c, d=d, sign_level=sign_level)
    ror_values, ror_lower, ror_upper = reporting_odds_ratio(a=a, b=b, c=c, d=d, sign_level=sign_level)
    ic_values, ic_lower, ic_upper = information_component(a=a, b=b, c=c, d=d, sign_level=sign_level, shrink=shrink)

    df = df.assign(prr=prr_values, prr_lower=prr_lower, prr_upper=prr_upper, ror=ror_values, ror_lower=ror_lower,
                   ror_upper=ror_upper, ic=ic_values, ic_lower=ic_lower, ic_upper=ic_upper)
    return df


def clean_metric_triplet(df: pd.DataFrame, metric: str):
    lower_col = f'{metric}_lower'
    upper_col = f'{metric}_upper'

    def clean_row(row):
        values = [row[metric], row[lower_col], row[upper_col]]
        if all(is_valid_float(val) for val in values):
            return row
        else:
            row[metric] = pd.NA
            row[lower_col] = pd.NA
            row[upper_col] = pd.NA
            return row

    return df.apply(clean_row, axis=1)


def metric_to_label(row, metric: str, min_records: int = 3):

    thresholds = {'prr': 1.0, 'ror': 1.0, 'ic': 0.0}
    threshold = thresholds.get(metric)

    if row['a'] < min_records:  # we cannot say how toxic a molecule is due to lack of data
        return 'Undefined'

    value_lower, value, value_upper = row[f'{metric}_lower'], row[f'{metric}'], row[f'{metric}_upper']

    if not all(is_valid_float(x) for x in [value, value_lower, value_upper]):  # check if all values are present and not missing
        return 'Ambiguous'

    if threshold <= value_lower:  # i.e. CI to the right of threshold
        return 'High'
    elif value_lower < threshold < value_upper:  # i.e. T within CI
        if threshold < value:
            return 'Moderate'
        elif threshold == value:
            return 'Ambiguous'
        elif value < threshold:
            return 'Low'
        else:
            raise ValueError('Check the setup')
    elif value_upper <= threshold:
        return 'Minimal'
    else:
        raise ValueError('Check the setup')


def conf_score(row, metric: str, dist_power: float = 1):

    thresholds = {'prr': 1.0, 'ror': 1.0, 'ic': 0.0}
    threshold = thresholds.get(metric)

    value_lower, value, value_upper = row[f'{metric}_lower'], row[f'{metric}'], row[f'{metric}_upper']
    label = row[f'{metric}_tox']

    if not all(is_valid_float(x) for x in [value, value_lower, value_upper]):  # check if all values are present and not missing
        return pd.NA

    if label in ['Ambiguous', 'Undefined']:
        return pd.NA
    elif label in ['High', 'Moderate', 'Low', 'Minimal']:
        ci_range = (value_upper - value_lower) + 0.0001
        distance = abs(value - threshold) ** dist_power
        confidence_score = np.round(distance / ci_range, 5)
        return confidence_score
    else:
        raise ValueError('Unknown label')


def mod_sigmoid(x, saturation: float = 2.5):
    """
    Modified sigmoid. Parameter m gives the position at which the function reaches 0.9.
    """
    if not is_valid_float(x):
        return pd.NA

    alpha = -2 / saturation * np.log(9)
    beta = saturation / 2

    value = 1 / (1 + np.exp(alpha * (x - beta)))
    return np.round(value, 5)


def step_9(df: pd.DataFrame, min_records: int = 3, sign_level: float = 0.05, shrink: float = 2.5, dist_power: float = 0.5,
           saturation: float = 2.5):
    """
    Calculate DPA metrics and assign toxicity labels
    """

    class_map = {
        'High': 1,
        'Moderate': 1,
        'Low': 0,
        'Minimal': 0
    }

    df = calculate_dpa_metrics(df, sign_level=sign_level, shrink=shrink)  # somehow this is vectorized ^_^
    df = df.astype({'prr': 'Float32', 'prr_lower': 'Float32', 'prr_upper': 'Float32',
                    'ror': 'Float32', 'ror_lower': 'Float32', 'ror_upper': 'Float32',
                    'ic': 'Float32', 'ic_lower': 'Float32', 'ic_upper': 'Float32'})

    for metric in ['prr', 'ror', 'ic']:
        df = clean_metric_triplet(df, metric)
        df.loc[:, f'{metric}_tox'] = pd.Series([metric_to_label(row, metric=metric, min_records=min_records) for idx, row in df.iterrows()])
        df.loc[:, f'{metric}_conf'] = pd.Series([conf_score(row, metric=metric, dist_power=dist_power) for idx, row in df.iterrows()])
        df.loc[:, f'{metric}_weight'] = df[f'{metric}_conf'].apply(mod_sigmoid, saturation=saturation)
        df.loc[:, f'{metric}_bin'] = df[f'{metric}_tox'].apply(lambda value: class_map.get(value))
        df = df.astype({f'{metric}_tox': 'string', f'{metric}_conf': 'Float32', f'{metric}_weight': 'Float32'})

    return df


"""
Step 10
"""


def smooth_labels(row, label_col: str, weight_col: str) -> np.ndarray:
    value = row[label_col]
    weight = row[weight_col]
    return np.round(np.array(weight * value + (1 - weight) * 0.5, dtype=np.float32), 5)


def step_10(df: pd.DataFrame, strat_type: str, pt_type: str, drop_unknown: bool = True, dpa_metric: str = 'ic'):
    """
    To be run within a single pt_type directory!
    """
    sex_order = CategoricalDtype(categories=['Male', 'Female', 'Unknown'], ordered=True)
    age_order = CategoricalDtype(categories=['Children', 'Adolescent', 'Adult', 'Elderly', 'Unknown'], ordered=True)
    wgt_order = CategoricalDtype(categories=['Low', 'Average', 'High', 'Unknown'], ordered=True)

    all_metrics = ['prr', 'ror', 'ic']
    all_metrics.pop(all_metrics.index(dpa_metric))

    for metric in all_metrics:
        df = df.drop(columns=[metric, f'{metric}_lower', f'{metric}_upper', f'{metric}_tox'])

    df = df.drop(columns=pt_type)

    strats = {'n': [],
              's': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    df = df[~df[f'{dpa_metric}_tox'].isin(['Undefined', 'Ambiguous'])]

    if drop_unknown and (strat_groups := strats.get(strat_type)) != 'n':
        for group in strat_groups:
            df = df[df[group] != 'Unknown'].copy()

    if 'Sex' not in df.columns:
        df = df.assign(Sex='Unknown')
    if 'Age' not in df.columns:
        df = df.assign(Age='Unknown')
    if 'Weight' not in df.columns:
        df = df.assign(Weight='Unknown')

    df['Strat'] = strat_type
    df[f'{dpa_metric}_smooth'] = df.apply(smooth_labels, label_col=f'{dpa_metric}_bin', weight_col=f'{dpa_metric}_weight', axis=1)
    df = df.astype({'Sex': sex_order, 'Age': age_order, 'Weight': wgt_order, 'Strat': 'string',
                    'a': 'UInt64', 'b': 'UInt64', 'c': 'UInt64', 'd': 'UInt64'})

    df = df.loc[:, ['Strat', 'Sex', 'Age', 'Weight', 'smi_enc', dpa_metric,
                    f'{dpa_metric}_lower', f'{dpa_metric}_upper', f'{dpa_metric}_tox',
                    f'{dpa_metric}_bin', f'{dpa_metric}_conf', f'{dpa_metric}_weight',
                    f'{dpa_metric}_smooth', 'a', 'b', 'c', 'd']]

    return df


"""
Step 11
Encoding
"""


def combine_arrays(row, columns: List[str]) -> np.ndarray:
    arrays = np.concatenate([row[col] for col in columns], axis=0)
    return arrays


def sex_to_array(value) -> np.ndarray:
    mapping = {'Male':    np.array([1, 0], dtype=np.int64),
               'Female':  np.array([0, 1], dtype=np.int64),
               'Unknown': np.array([0, 0], dtype=np.int64)}
    return mapping.get(value, np.nan)


def age_to_array(value) -> np.ndarray:
    mapping = {'Children':   np.array([1, 0, 0, 0], dtype=np.int64),
               'Adolescent': np.array([0, 1, 0, 0], dtype=np.int64),
               'Adult':      np.array([0, 0, 1, 0], dtype=np.int64),
               'Elderly':    np.array([0, 0, 0, 1], dtype=np.int64),
               'Unknown':    np.array([0, 0, 0, 0], dtype=np.int64)}
    return mapping.get(value, np.nan)


def wgt_to_array(value) -> np.ndarray:
    mapping = {'Low':     np.array([1, 0, 0], dtype=np.int64),
               'Average': np.array([0, 1, 0], dtype=np.int64),
               'High':    np.array([0, 0, 1], dtype=np.int64),
               'Unknown': np.array([0, 0, 0], dtype=np.int64)}
    return mapping.get(value, np.nan)


def pack_to_tuple(row, cols: List[str]):
    return tuple([row[col] for col in cols])


def pack_to_array(row, cols: List[str]):
    return np.array([row[col] for col in cols])


def step_11(df: pd.DataFrame, smiles_col: str, idx_2_smi: dict, dpa_metric: str) -> pd.DataFrame:

    df.loc[:, 'SMILES'] = [idx_2_smi.get(idx) for idx in df[smiles_col]]
    df['Sex_enc'] = [sex_to_array(value) for value in df['Sex']]
    df['Age_enc'] = [age_to_array(value) for value in df['Age']]
    df['Weight_enc'] = [wgt_to_array(value) for value in df['Weight']]
    df['Demo'] = df.apply(combine_arrays, columns=['Sex_enc', 'Age_enc', 'Weight_enc'], axis=1)

    df['Signature'] = df.apply(pack_to_tuple, cols=['Sex', 'Age', 'Weight'], axis=1)
    df['DPA_confusion'] = df.apply(pack_to_array, cols=['a', 'b', 'c', 'd'], axis=1)
    df['DPA_values'] = df.apply(pack_to_array, cols=[dpa_metric, f'{dpa_metric}_lower', f'{dpa_metric}_upper'], axis=1)
    df['DPA_values'] = df['DPA_values'].apply(lambda array: np.round(array.reshape(-1), 5))

    df = df.rename(columns={
        'Strat': 'Stratification',
        f'{dpa_metric}_tox': 'Risk',
        f'{dpa_metric}_bin': 'Label',
        f'{dpa_metric}_conf': 'Label_confidence',
        f'{dpa_metric}_weight': 'Label_weight',
        f'{dpa_metric}_smooth': 'Label_regression',
        'Demo': 'DemoFP'
    })

    df = df.drop(columns=['Sex', 'Age', 'Weight', 'a', 'b', 'c', 'd', 'smi_enc', 'Sex_enc', 'Age_enc', 'Weight_enc'])

    df = df.astype({
        'Stratification': 'string',
        'Risk': 'string',
        'Label': 'UInt8',
        'Label_confidence': 'Float32',
        'Label_weight': 'Float32',
        'Label_regression': 'Float32',
        'SMILES': 'string',
    })

    df = df.loc[:, ['SMILES', 'Signature', 'Risk', 'Label', 'Label_confidence', 'Label_weight', 'Label_regression',
                    'DPA_confusion', 'DPA_values', 'DemoFP', 'Stratification']]

    return df


def step_12(df: pd.DataFrame, fold_df: pd.DataFrame, smiles_col: str, dataset_type:str, dpa_type: str, pt_set: str) -> pd.DataFrame:

    df.attrs = {
        'Dataset': dataset_type,
        'Metric': dpa_type,
        'Set': pt_set
    }

    df = df.merge(fold_df, on=smiles_col, how='left')

    if df.Cluster_ID.isna().sum() != 0:
        print(f'Some compounds in {df.attrs} have not been assigned to a cluster')
    if df.Fold.isna().sum() != 0:
        print(f'Some compounds in {df.attrs} have not been assigned to a fold')

    df = df.astype({'SMILES': 'string', 'Fold': 'UInt8', 'Cluster_ID': 'UInt16'})
    return df

