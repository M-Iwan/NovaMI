import math
from itertools import chain
from typing import Union, List
from joblib import Parallel, delayed

import numpy as np
import numpy.typing as npt
import polars as pl
from rdkit import Chem, RDLogger


def validate_smiles(smiles: Union[str, List[str], npt.NDArray[np.str_]]):
    """
    Check validity of SMILES and identify existing problems.

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
            return f"Wrong type: {type(smi)}"
        if smi == "":
            return "Empty string"
        try:
            if Chem.MolFromSmiles(smi, sanitize=True) is not None:
                return ""
            if (mol := Chem.MolFromSmiles(smi, sanitize=False)) is None:
                return "Invalid SMILES"
            issues = [
                str(problem.GetType()) for problem in Chem.DetectChemistryProblems(mol)
            ]
            return " | ".join(issues)
        except Exception as e:
            print(f"Unexpected error:\n{e}")
            return f"Error: {e}"

    if isinstance(smiles, str):
        return process_smiles(smiles)

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        return [
            process_smiles(smi) for smi in smiles
        ]
    else:
        raise TypeError(f"Expected smiles to be one of str, List[str], np.ndarray[str] got {type(smiles)} instead")


def validate(df: pl.DataFrame, smiles_col: str = "SMILES", out_col: str = "Issues",
             n_jobs: int = 1, batch_size: int = 512, timeout: int = 600):
    """
    Validate and identify potential issues with SMILES in a polars DataFrame.

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
        return df.with_columns(pl.lit(None).alias(out_col))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    results = Parallel(n_jobs=n_jobs, verbose=1, timeout=timeout, backend="loky")(
        delayed(validate_smiles)(smiles=smi) for smi in smiles_batches
    )

    issues = list(chain.from_iterable(results))

    smiles_df = pl.DataFrame({
        "_smiles_key": smiles,
        "_issues": issues
    })

    df = df.join(smiles_df, left_on=smiles_col, right_on="_smiles_key", how="left")

    df = df.with_columns([
        pl.col("_issues").alias(out_col)
    ]).drop("_issues")

    return df