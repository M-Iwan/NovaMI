import math
from itertools import chain
from typing import Union, List
from joblib import Parallel, delayed
import numpy as np
import numpy.typing as npt
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


def filter_inorganic_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Remove SMILES with no carbon atom (purely inorganic molecules).

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')
    carbon_pattern = Chem.MolFromSmarts('[#6]')  # checks if a molecule contains at least one carbon atom

    def process_smiles(smi, pattern):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            if not mol.HasSubstructMatch(pattern):
                print(f'No carbon atom found in < {smi} >')
                return None
            return smi
        except Exception as e:
            print(f'Could not filter < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            pattern=carbon_pattern
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                pattern=carbon_pattern
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def filter_inorganic(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "Filtered",
                     n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Remove SMILES with no carbon atom (purely inorganic molecules) in a polars DataFrame.

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
            pl.lit(None, dtype=pl.Utf8).alias(out_col),
        )

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(filter_inorganic_smiles)(smiles=smi) for smi in smiles_batches
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


def filter_uncommon_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Remove SMILES containing elements rarely found in pharmaceutical/drug-like substances.

    Allowed elements cover common organic atoms, halogens, and metals/metalloids
    seen in approved drugs and pharmaceutical excipients/salts/contrast agents
    (e.g. Li, Na, K, Ca, Mg, Fe, Co, Cu, Zn, As, Se, Mo, Tc, Ag, Sn, Sb, I, Gd,
    Pt, Au, Bi).

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')
    # H, Li, B, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca, Mn, Fe, Co, Cu, Zn,
    # As, Se, Br, Mo, Tc, Ag, Sn, Sb, I, Gd, Pt, Au, Bi
    allowed_atomic_nums = [
        1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16,
        17, 19, 20, 25, 26, 27, 29, 30, 33, 34, 35, 42,
        43, 47, 50, 51, 53, 64, 78, 79, 83
    ]
    smarts = '[' + ';'.join(f'!#{n}' for n in allowed_atomic_nums) + ']'
    uncommon_pattern = Chem.MolFromSmarts(smarts)

    def process_smiles(smi, pattern):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            if mol.HasSubstructMatch(pattern):
                print(f'Uncommon element found in < {smi} >')
                return None
            return smi
        except Exception as e:
            print(f'Could not filter < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(
            smi=smiles,
            pattern=uncommon_pattern
        )

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(
                smi=smi,
                pattern=uncommon_pattern
            ) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def filter_uncommon(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "Filtered",
                    n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Remove SMILES containing elements rarely found in pharmaceutical/drug-like
    substances, in a polars DataFrame.

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
        delayed(filter_uncommon_smiles)(smiles=smi) for smi in smiles_batches
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


def mark_uncommon_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Flag SMILES containing elements rarely found in pharmaceutical/drug-like substances.

    Allowed elements cover common organic atoms, halogens, and metals/metalloids
    seen in approved drugs and pharmaceutical excipients/salts/contrast agents
    (e.g. Li, Na, K, Ca, Mg, Fe, Co, Cu, Zn, As, Se, Mo, Tc, Ag, Sn, Sb, I, Gd,
    Pt, Au, Bi).

    Parameters
    ----------
    smiles: Union[str, List[str], npt.NDArray[np.str_]]

    Returns
    -------
    smiles: Union[str, List[str]]
    """
    RDLogger.DisableLog('rdApp.*')

    allowed_atomic_nums = {
        1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16,
        17, 19, 20, 25, 26, 27, 29, 30, 33, 34, 35, 42,
        43, 47, 50, 51, 53, 64, 78, 79, 83
    }

    def process_smiles(smi: str):
        if not isinstance(smi, str):
            print(f"Expected smiles to be string, got {type(smi)} instead.")
            return None
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None
        try:
            uncommon = sorted(
                {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() not in allowed_atomic_nums}
            )
            return " | ".join(uncommon)
        except Exception as e:
            print(f'Could not process < {smi} > due to \n{e}')
            return None

    if isinstance(smiles, str):
        return process_smiles(smiles)

    elif isinstance(smiles, (list, np.ndarray)):
        return [process_smiles(smi) for smi in smiles]

    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def mark_uncommon(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "uncommon_elements",
                  n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Flag rows whose SMILES contain elements rarely found in pharmaceutical/drug-like .substances

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
        delayed(mark_uncommon_smiles)(smiles=batch) for batch in smiles_batches
    )

    flags = list(chain.from_iterable(results))

    flags_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_flag": flags,
    })

    df = (df
        .join(flags_df, left_on=smiles_col, right_on="_smiles_key", how="left")
        .with_columns(pl.col("_flag").alias(out_col))
        .drop("_flag")
    )

    return df


def filter_properties_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Remove SMILES with properties outside predefined ranges. Values are based on
    the distributions of standardized ChEMBL (v. 36) compounds and are intended
    to remove the most extreme cases only.

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
            print(f'Expected smiles to be of type str, got {type(smi)} instead')
            return None, f"Invalid type: {type(smi)}"
        if (mol := Chem.MolFromSmiles(smi)) is None:
            print(f'Unable to construct a valid molecule from < {smi} >')
            return None, f"Invalid SMILES"
        try:
            mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
            log_p, mol_r = rdMolDescriptors.CalcCrippenDescriptors(mol)
            num_heavy = rdMolDescriptors.CalcNumHeavyAtoms(mol)
            num_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            n_hba = rdMolDescriptors.CalcNumHBA(mol)
            n_hbd = rdMolDescriptors.CalcNumHBD(mol)
            n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
            n_rings = rdMolDescriptors.CalcNumRings(mol)

            bounds = [
                ("MolWt", mol_wt, 32, 2048),
                ("LogP", log_p, -10, 12),
                ("MolMR", mol_r, 0, 512),
                ("HeavyAtoms", num_heavy, 2, 128),
                ("Heteroatoms", num_hetero, 0, 48),
                ("TPSA", tpsa, 0, 512),
                ("HBA", n_hba, 0, 24),
                ("HBD", n_hbd, 0, 16),
                ("RotatableBonds", n_rot, 0, 32),
                ("Rings", n_rings, 0, 12),
            ]

            violations = []
            for name, value, low, high in bounds:
                if value < low:
                    violations.append(f'{name}: {value:.2f} < {low}')
                elif value > high:
                    violations.append(f'{name}: {value:.2f} > {high}')

            if violations:
                return None, " | ".join(violations)
            return smi, None

        except Exception as e:
            print(f"Could not filter < {smi} > due to \n{e}")
            return None, f"Error:{e}"

    if isinstance(smiles, str):
        return process_smiles(smiles)

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(smi) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], npt.NDArray[np.str_] got {type(smiles)} instead")


def filter_properties(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "Filtered",
                      n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Remove SMILES with properties outside predefined ranges, in a polars DataFrame.

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
            pl.lit(None, dtype=pl.Utf8).alias("_property_violation"),
        ])

    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(filter_properties_smiles)(smiles=smi) for smi in smiles_batches
    )

    results = list(chain.from_iterable(results))
    filtered_smiles = [result[0] for result in results]
    violations = [result[1] for result in results]

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_new_smiles": filtered_smiles,
        "_new_violations": violations
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns([
        pl.col("_new_smiles").alias(out_col),
        pl.col("_new_violations").alias("_property_violation")
    ]).drop(["_new_smiles", "_new_violations"])

    return df