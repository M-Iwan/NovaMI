"""
File with functions related to dataset preparation and filtering.
"""
import math
from itertools import chain
from typing import Union, List
from joblib import Parallel, delayed

import numpy as np
import numpy.typing as npt
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover, RemoveStereochemistry
from rdkit.Chem.MolStandardize import rdMolStandardize


def normalize_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Normalize functional group representations in molecules

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')
    normalizer = rdMolStandardize.Normalizer()

    def process_smiles(smi, normalizer):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            mol = normalizer.normalize(mol)
            if mol is None:
                print(f"Could not normalize < {smi} >")
                return None
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f'Could not normalize < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            normalizer=normalizer
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                normalizer=normalizer
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def normalize(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "CanSMILES",
              n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Normalize functional group representations for SMILES in a polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with SMILES
    smiles_col: str
        Name of a column holding SMILES
    out_col: str
        Name of the output column
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    df: pl.DataFrame
        Updated Polars DataFrame
    """
    smiles = list(set(df[smiles_col].drop_nulls().to_list()))

    if len(smiles) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(out_col)
        )

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(normalize_smiles)(smiles=smi) for smi in smiles_batches
    )

    can_smiles = list(chain.from_iterable(results))

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": can_smiles
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns(
        pl.col("_new_smiles").alias(out_col)
    ).drop("_new_smiles")

    return df


def disconnect_metals_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Disconnect metals from molecules

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')
    metal_disconnector = rdMolStandardize.MetalDisconnector()

    def process_smiles(smi, disconnector):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            mol = disconnector.Disconnect(mol)
            if mol is None:
                print(f"Could not disconnect metals for < {smi} >")
                return None
            mol = rdMolStandardize.DisconnectOrganometallics(mol)
            if mol is None:
                print(f"Could not disconnect organometallics for < {smi} >")
                return None
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f'Could not disconnect metals from < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            disconnector=metal_disconnector
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                disconnector=metal_disconnector
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def disconnect_metals(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "CanSMILES",
                      n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Disconnect metals from SMILES in a polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with SMILES
    smiles_col: str
        Name of a column holding SMILES
    out_col: str
        Name of the output column
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    df: pl.DataFrame
        Updated Polars DataFrame
    """
    smiles = list(set(df[smiles_col].drop_nulls().to_list()))

    if len(smiles) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(out_col)
        )

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(disconnect_metals_smiles)(smiles=smi) for smi in smiles_batches
    )

    can_smiles = list(chain.from_iterable(results))

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": can_smiles
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns(
        pl.col("_new_smiles").alias(out_col)
    ).drop( "_new_smiles")

    return df


def strip_salts_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Strip common salts from SMILES

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')
    salt_remover = SaltRemover.SaltRemover()

    def process_smiles(smi, remover):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None, None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None, None
        try:
            mol, deleted_mol = remover.StripMolWithDeleted(mol, dontRemoveEverything=True)
            dels = [Chem.MolToSmarts(del_mol) for del_mol in deleted_mol]
            if mol is None:
                print(f"Could not strip salts for < {smi} >")
                return None, None
            if dels:
                return Chem.MolToSmiles(mol), " | ".join(dels)
            else:
                return Chem.MolToSmiles(mol), None
        except Exception as e:
            print(f'Could not strip salts from < {smi} > due to \n{e}')
            return None, None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            remover=salt_remover
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                remover=salt_remover
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def strip_salts(df: pl.DataFrame, smiles_col: str = "SMILES", out_col = "CanSMILES",
                n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Strip common salts from SMILES in a polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with SMILES
    smiles_col: str
        Name of a column holding SMILES
    out_col: str
        Name of the output column
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    df: pl.DataFrame
        Updated Polars DataFrame
    """

    smiles = list(set(df[smiles_col].drop_nulls().to_list()))

    if len(smiles) == 0:
        return df.with_columns([
            pl.lit(None, dtype=pl.Utf8).alias(out_col),
            pl.lit(None, dtype=pl.Utf8).alias("_stripped_salts")
        ])

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(strip_salts_smiles)(smiles=smi) for smi in smiles_batches
    )

    results = list(chain.from_iterable(results))
    stripped_smiles = [result[0] for result in results]
    removed_salts = [result[1] for result in results]

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": stripped_smiles,
        "_new_salts": removed_salts
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns([
        pl.col("_new_smiles").alias(out_col),
        pl.col("_new_salts").alias("_stripped_salts")
    ]).drop(["_new_smiles", "_new_salts"])

    return df


def remove_minor_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Remove minor fragments from molecules

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """

    RDLogger.DisableLog('rdApp.*')
    fragment_remover = rdMolStandardize.LargestFragmentChooser(
        preferOrganic=True
    )

    def process_smiles(smi, remover):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            mol = remover.choose(mol)
            if mol is None:
                print(f"Could not remove minor fragments for < {smi} >")
                return None
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f'Could not remove minor fragments for < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            remover=fragment_remover
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                remover=fragment_remover
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def remove_minor(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "CanSMILES",
                 n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Remove minor fragments from SMILES in a polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with SMILES
    smiles_col: str
        Name of a column holding SMILES
    out_col: str
        Name of the output column
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    df: pl.DataFrame
        Updated Polars DataFrame
    """
    smiles = list(set(df[smiles_col].drop_nulls().to_list()))

    if len(smiles) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(out_col)
        )

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(remove_minor_smiles)(smiles=smi) for smi in smiles_batches
    )

    can_smiles = list(chain.from_iterable(results))

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": can_smiles
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns(
        pl.col("_new_smiles").alias(out_col)
    ).drop("_new_smiles")

    return df


def remove_stereochemistry_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Remove stereochemistry from molecules

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """

    RDLogger.DisableLog('rdApp.*')

    def process_smiles(smi):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            RemoveStereochemistry(mol)  # in-place modification
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f'Could not remove stereochemistry for < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(smiles)

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(smi) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def remove_stereochemistry(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "CanSMILES",
                           n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Remove stereochemistry from SMILES in a polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with SMILES
    smiles_col: str
        Name of a column holding SMILES
    out_col: str
        Name of the output column
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    df: pl.DataFrame
        Updated Polars DataFrame
    """
    smiles = list(set(df[smiles_col].drop_nulls().to_list()))

    if len(smiles) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(out_col)
        )

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(remove_stereochemistry_smiles)(smiles=smi) for smi in smiles_batches
    )

    can_smiles = list(chain.from_iterable(results))

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": can_smiles
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns(
        pl.col("_new_smiles").alias(out_col)
    ).drop("_new_smiles")

    return df


def remove_charges_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Remove charges from molecules

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')
    reionizer = rdMolStandardize.Reionizer()
    uncharger = rdMolStandardize.Uncharger()

    def process_smiles(smi, reionizer, uncharger):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            mol = reionizer.reionize(mol)
            if mol is None:
                print(f"Could not remove charges for < {smi} >")
                return None
            mol = uncharger.uncharge(mol)
            if mol is None:
                print(f"Could not remove charges for < {smi} >")
                return None
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f'Could not remove charges for < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            reionizer=reionizer,
            uncharger=uncharger
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                reionizer=reionizer,
                uncharger=uncharger
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def remove_charges(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "CanSMILES",
                   n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Remove charges from SMILES in a polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with SMILES
    smiles_col: str
        Name of a column holding SMILES
    out_col: str
        Name of the output column
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    df: pl.DataFrame
        Updated Polars DataFrame
    """
    smiles = list(set(df[smiles_col].drop_nulls().to_list()))

    if len(smiles) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(out_col)
        )

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(remove_charges_smiles)(smiles=smi) for smi in smiles_batches
    )

    can_smiles = list(chain.from_iterable(results))

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": can_smiles
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns(
        pl.col("_new_smiles").alias(out_col)
    ).drop("_new_smiles")

    return df


def enumerate_tautomers_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]], max_num_tautomers: int = 32):
    """
    Select canonical tautomer for SMILES

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]
        SMILES to process
    max_num_tautomers : int
        Number of tautomers to evaluate
    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')

    tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
    tautomer_enumerator.SetMaxTautomers(max_num_tautomers)

    def process_smiles(smi, enumerator):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            mol = enumerator.Canonicalize(mol)
            if mol is None:
                print(f"Could not select canonical tautomer for < {smi} >")
                return None
            return Chem.MolToSmiles(mol)
        except Exception as e:
            print(f'Could not select canonical tautomer for < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            enumerator=tautomer_enumerator
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                enumerator=tautomer_enumerator
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def enumerate_tautomers(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "CanSMILES",
                        max_num_tautomers: int = 32, n_jobs: int = 1, batch_size: int = 512, timeout: int = 1200):
    """
    Select canonical tautomers for SMILES in a polars DataFrame.

    Parameters
    ----------
    df: pl.DataFrame
        Polars DataFrame with SMILES
    smiles_col: str
        Name of a column holding SMILES
    out_col: str
        Name of the output column
    max_num_tautomers : int
        Number of tautomers to evaluate
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    df: pl.DataFrame
        Updated Polars DataFrame
    """
    smiles = list(set(df[smiles_col].drop_nulls().to_list()))

    if len(smiles) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(out_col)
        )

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(enumerate_tautomers_smiles)(smiles=smi, max_num_tautomers=max_num_tautomers) for smi in smiles_batches
    )

    can_smiles = list(chain.from_iterable(results))

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": can_smiles
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns(
        pl.col("_new_smiles").alias(out_col)
    ).drop("_new_smiles")

    return df


def pipeline_clean(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "CanSMILES",
                    n_jobs: int = 1, batch_size: int = 512, max_num_tautomers: int = 16, timeout: int = 1200) -> pl.DataFrame:
    """
    Process a chemical compound DataFrame through a standardization pipeline.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing chemical structures. Must include a column
        that can be processed as SMILES strings.
    smiles_col : str
        Name of the column holding the raw input SMILES.
    out_col : str
        Name of the column that will hold the standardized SMILES.
    n_jobs : int, optional
        Number of cores to use for calculations.
    batch_size : int, optional
        Number of SMILES per batch.
    max_num_tautomers : int
        Number of tautomers to evaluate in the final step.
    timeout: int
        Timeout parameter for Parallel computation

    Returns
    -------
    pl.DataFrame
        Processed DataFrame with standardized chemical structures.

    Notes
    -----
    The pipeline performs the following operations in order:
    1. Normalizes functional groups
    2. Disconnects metals from structures
    3. Removes salt counterions
    4. Selects parent compounds (largest fragment, fallback for anything step 2 missed)
    5. Removes stereochemistry information (flattens compounds)
    6. Neutralizes formal charges
    7. Selects canonical tautomer
    """

    n_initial = df.height
    print(f'> Starting standardization pipeline on {n_initial} rows')

    def _log(df, step_name):
        n_null = df.filter(pl.col(out_col).is_null()).height
        print(f'  [{step_name}] {n_initial - n_null}/{n_initial} rows have a valid {out_col} '
              f'({n_null} null)')

    print('> Normalizing functional groups')
    df = normalize(df, smiles_col=smiles_col, out_col=out_col, n_jobs=n_jobs,
                   batch_size=batch_size, timeout=timeout)
    _log(df, 'normalization')

    print('> Disconnecting metals')
    df = disconnect_metals(df, smiles_col=out_col, out_col=out_col, n_jobs=n_jobs,
                           batch_size=batch_size, timeout=timeout)
    _log(df, 'metals')

    print('> Stripping salts')
    df = strip_salts(df, smiles_col=out_col, out_col=out_col, n_jobs=n_jobs,
                     batch_size=batch_size, timeout=timeout)
    _log(df, 'salts')

    print('> Selecting parent compounds')
    df = remove_minor(df, smiles_col=out_col, out_col=out_col, n_jobs=n_jobs,
                      batch_size=batch_size, timeout=timeout)
    _log(df, 'largest fragment')

    print('> Flattening compounds')
    df = remove_stereochemistry(df, smiles_col=out_col, out_col=out_col, n_jobs=n_jobs,
                                batch_size=batch_size, timeout=timeout)
    _log(df, 'stereochemistry')

    print('> Neutralizing charges')
    df = remove_charges(df, smiles_col=out_col, out_col=out_col, n_jobs=n_jobs,
                        batch_size=batch_size, timeout=timeout)
    _log(df, 'charges')

    print('> Selecting canonical tautomer')
    df = enumerate_tautomers(df, smiles_col=out_col, out_col=out_col, max_num_tautomers=max_num_tautomers,
                             n_jobs=n_jobs, batch_size=batch_size, timeout=timeout)
    _log(df, 'tautomers')

    n_dropped = df.filter(pl.col(out_col).is_null()).height
    print(f'> Pipeline complete: {n_initial - n_dropped}/{n_initial} rows standardized '
          f'({n_dropped} could not be resolved)')

    return df
