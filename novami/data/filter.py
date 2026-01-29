from typing import Dict

import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors


def filter_outliers(df: pl.DataFrame, smiles_col: str = 'SMILES', kwargs: Dict = None):
    """
    Function that applies very light filtering to provided SMILES with the aim of removing
    obvious outliers.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame containing SMILES strings.
    smiles_col: str
        Name of the column holding SMILES strings.
    kwargs: Dict
        Optional updates to default filtering values.

    Returns
    -------
    df: pl.DataFrame
    """

    default_filters = {
        'MolWt': (30, 1200),
        'MolLogP': (-7, 9),
        'MolMR': (0, 400),
        'TPSA': (0, 200),
        'NumHeavyAtom': (1, 100),
        'NumHAcceptors': (0, 24),
        'NumHDonors': (0, 12),
        'NumHeteroatoms': (0, 24),
        'NumRotatableBonds': (0, 32),
        'NumAliphaticRings': (0, 10),
        'NumAromaticRings': (0, 10)
    }

    if kwargs is not None:
        default_filters.update(kwargs)

    initial_columns = df.columns

    df = df.with_columns(
        pl.col(smiles_col).map_elements(Chem.MolFromSmiles).alias('Mol')
    )

    df = df.with_columns([
        pl.col('Mol').map_elements(Descriptors.MolWt, return_dtype=pl.Float32).alias('MolWt'),
        pl.col('Mol').map_elements(Descriptors.MolLogP, return_dtype=pl.Float32).alias('MolLogP'),
        pl.col('Mol').map_elements(Descriptors.MolMR, return_dtype=pl.Float32).alias('MolMR'),
        pl.col('Mol').map_elements(Descriptors.TPSA, return_dtype=pl.Float32).alias('TPSA'),
        pl.col('Mol').map_elements(Descriptors.HeavyAtomCount, return_dtype=pl.Int16).alias('NumHeavyAtom'),
        pl.col('Mol').map_elements(Descriptors.NumHAcceptors, return_dtype=pl.Int16).alias('NumHAcceptors'),
        pl.col('Mol').map_elements(Descriptors.NumHDonors, return_dtype=pl.Int16).alias('NumHDonors'),
        pl.col('Mol').map_elements(Descriptors.NumHeteroatoms, return_dtype=pl.Int16).alias('NumHeteroatoms'),
        pl.col('Mol').map_elements(Descriptors.NumRotatableBonds, return_dtype=pl.Int16).alias('NumRotatableBonds'),
        pl.col('Mol').map_elements(Descriptors.NumAliphaticRings, return_dtype=pl.Int16).alias('NumAliphaticRings'),
        pl.col('Mol').map_elements(Descriptors.NumAromaticRings, return_dtype=pl.Int16).alias('NumAromaticRings'),
    ])

    for key, filter_range in default_filters.items():
        df = df.filter(pl.col(key).is_between(*filter_range))

    df = df[initial_columns]

    return df