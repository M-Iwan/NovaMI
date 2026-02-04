import json
import os
import pickle
import requests
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdFingerprintGenerator

import torch

from novami.io.file import read_pd, write_pd


def smiles_2_morgan(smiles: Union[str, List[str]], radius: int = 2, nbits: int = 1024, dense: bool = True, mfpgen=None):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    radius: int, optional
        The radius parameter for Morgan FP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is .
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.
    mfpgen: rdkit.Chem.rdFingerprintGenerator.FingeprintGenerator64, optional
        If passed, the radius and nbits will be ignored.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """

    if mfpgen is None:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    def get_embedding(smi: str):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.array(mfpgen.GetFingerprint(mol), dtype=np.uint8)

            return fp if dense else fp.nonzero()[0]

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_morgan(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', descriptor_col: str = 'Morgan',
                       radius: int = 2, nbits: int = 1024, dense: bool = True):
    """
    Convert SMILES in a DataFrame to Morgan fingerprints.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    radius: int, optional
        The radius parameter for Morgan FP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.

    Returns
    -------
    df : pd.DataFrame, pl.DataFrame
        A pandas/polars Dataframe with added Morgan fingerprints.
        If 'dense' is set to False, returns a numpy array with indices of non-zero elements.
    """

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    if isinstance(df, pd.DataFrame):
        df[descriptor_col] = smiles_2_morgan(df[smiles_col].tolist(), radius=radius, nbits=nbits, dense=dense, mfpgen=mfpgen)
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series(descriptor_col, smiles_2_morgan(df[smiles_col].to_list(),
                                       radius=radius, nbits=nbits, dense=dense, mfpgen=mfpgen)))
        return df
    else:
        raise TypeError(f"Expected df to be either polars or pandas DataFrame, got {type(df)} instead.")


def smiles_2_maccs(smiles: Union[str, List[str]], dense: bool = True):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """

    def get_embedding(smi: str):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.array(rdMolDescriptors.GetMACCSKeysFingerprint(mol), dtype=np.uint8)

            return fp if dense else fp.nonzero()[0]

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_maccs(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', descriptor_col: str = 'MACCS',
                      dense: bool = True):
    """
    Convert SMILES in dataframe to MACCS fingerprints.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added MACCS fingerprints for given SMILES.
        If 'dense' is set to False, returns a numpy array with indices of non-zero elements.
    """

    if isinstance(df, pd.DataFrame):
        df[descriptor_col] = smiles_2_maccs(df[smiles_col].tolist(), dense=dense)
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series(descriptor_col, smiles_2_maccs(df[smiles_col].to_list(), dense=dense)))
        return df
    else:
        raise TypeError(f"Expected df to be either polars or pandas DataFrame, got {type(df)} instead.")


def smiles_2_klek(smiles: Union[str, List[str]], dense: bool = True, klek_mols: List = None,
                  source: str = 'src/src_files/klek.pkl'):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.
    klek_mols: List
        A list of RDKit molecules from Klekota&Roth SMARTS definitions.
        If not passed, they are read from source parameter.
    source: str
        A path to pickled List of klek_mols

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """

    if klek_mols is None:
        klek_mols = pickle.load(open(source, 'rb'))

    def get_embedding(smi: str, kmols: List):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.array([1 if mol.HasSubstructMatch(km) else 0 for km in kmols], dtype=np.uint8)

            return fp if dense else fp.nonzero()[0]

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles, klek_mols)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi, klek_mols) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_klek(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', descriptor_col: str = 'Klek',
                     dense: bool = True, source: str = 'src/src_files/klek.pkl'):
    """
    Convert SMILES in dataframe to Klekota&Roth fingerprints.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.
    source: str
        A path to pickled List of RDKit molecules generated from Klekota&Roth SMARTS.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added Klekota&Roth fingerprints.
        If 'dense' is set to False, returns a numpy array with indices of non-zero elements.
    """

    klek_mols = pickle.load(open(source, 'rb'))

    if isinstance(df, pd.DataFrame):
        df[descriptor_col] = smiles_2_klek(df[smiles_col].tolist(), dense=dense, klek_mols=klek_mols)
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series(descriptor_col, smiles_2_klek(df[smiles_col].to_list(), dense=dense)))
        return df
    else:
        raise TypeError(f"Expected df to be either polars or pandas DataFrame, got {type(df)} instead.")


def smiles_2_rdkit(smiles: Union[str, List[str]], decimals: int = 5):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """

    def get_embedding(smi: str):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.round(np.fromiter(Descriptors.CalcMolDescriptors(mol, silent=False, missingVal=np.nan).values(), dtype=np.float64), decimals)
            return fp

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_rdkit(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', descriptor_col: str = 'RDKit',
                      decimals: int = 5):
    """
    Convert SMILES in dataframe to RDKit descriptors.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added RDKit descriptors.
        If 'dense' is set to False, returns a numpy array with indices of non-zero elements.
    """

    if isinstance(df, pd.DataFrame):
        df[descriptor_col] = smiles_2_rdkit(df[smiles_col].tolist(), decimals=decimals)
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series(descriptor_col, smiles_2_rdkit(df[smiles_col].to_list(), decimals=decimals)))
        return df
    else:
        raise TypeError(f"Expected df to be either polars or pandas DataFrame, got {type(df)} instead.")


def smiles_2_chemberta(smiles: Union[str, List[str]], decimals: int = 5):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """
    try:
        from transformers import AutoTokenizer, AutoModel, logging
    except ImportError:
        raise ImportError("Function < smiles_2_chemberta > requires < transformers > library. Please install it "
                          "with < pip install transformers >")

    logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-100M-MLM')
    model = AutoModel.from_pretrained('DeepChem/ChemBERTa-100M-MLM')
    model.eval()

    def get_embeddings(smi: str):

        try:
            tokens = tokenizer(smi, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                output = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
            return np.round(output, decimals)

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embeddings(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embeddings(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_chemberta(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', descriptor_col: str = 'ChemBERTa', decimals: int = 5):
    """
    Convert SMILES in pandas/polars DataFrame to ChemBERTa embeddings.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    smiles_col : str
        Name of column with SMILES.
    descriptor_col : str
        Name of column to which add calculated descriptors.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added column holding CDDD descriptors for given SMILES.
    """
    try:
        from transformers import AutoTokenizer, AutoModel, logging
    except ImportError:
        raise ImportError("Function < dataframe_2_chemberta > requires < transformers > library. Please install it "
                          "with < pip install transformers >")

    if isinstance(df, pd.DataFrame):
        df[descriptor_col] = smiles_2_chemberta(df[smiles_col].tolist(), decimals=decimals)
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series(descriptor_col, smiles_2_chemberta(df[smiles_col].to_list(), decimals=decimals)))
        return df
    else:
        raise TypeError(f"Expected df to be either polars or pandas DataFrame, got {type(df)} instead.")


def dataframe_2_mordred(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', desc_col: str = 'Mordred',
                        path_source: str = f'src/src_files/mordred_paths.json', decimals: int = 5):
    """
    Convert SMILES in dataframe to Mordred descriptors.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    smiles_col : str
        Name of column with SMILES.
    desc_col : str
        Name of column to which add calculated descriptors.
    path_source : str
        Path to mordred_paths.json file.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added column holding Mordred descriptors for given SMILES.
    """

    def postprocess(entry, decimals_):
        if not isinstance(entry, (np.ndarray, list)):
            return np.nan
        if isinstance(entry, np.ndarray):
            return np.round(entry.reshape(-1), decimals_)
        if isinstance(entry, list):
            return [np.round(array.reshape(-1), decimals_) for array in entry]
        else:
            raise TypeError(f"Expected numpy array or list, got {type(entry)} instead.")

    with open(path_source, 'r') as file:
        paths = json.load(file)

    command = (f"{paths['python']} {paths['wrapper']} --input {paths['input']} --output {paths['output']} "
               f"--smiles_col {smiles_col} --descriptor_col {desc_col}")

    if is_polars := isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # pack the lists to strings so that they don't get broken -,-
    df.loc[:, smiles_col] = df[smiles_col].apply(lambda entry: ' : '.join(entry) if isinstance(entry, list) else entry)
    write_pd(df, paths['input'])

    os.system(command)

    df = read_pd(paths['output'])
    df.loc[:, desc_col] = df[desc_col].apply(postprocess, decimals_=decimals)

    if is_polars:
        df = pl.from_pandas(df)

    return df


def dataframe_2_cddd(df: Union[pd.DataFrame, pl.DataFrame], cddd_paths: str, smiles_col: str = 'SMILES',
                     descriptor_col: str = 'CDDD', n_cpus: int = 4, decimals: int = 5):
    """
    Convert SMILES in a DataFrame to CDDD descriptors.

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    cddd_paths : str
        Path to a JSON file containing paths to CDDD-related files. Generated using CDDD_conf.sh
    smiles_col : str
        Name of column with SMILES.
    descriptor_col : str
        Name of column to which add calculated descriptors.
    n_cpus : int
        Number of CPUs to use during processing
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added column holding CDDD descriptors for given SMILES.
    """

    def postprocess(entry, decimals_):
        if not isinstance(entry, (np.ndarray, list)):
            return np.nan
        if isinstance(entry, np.ndarray):
            return np.round(entry.reshape(-1), decimals_)
        if isinstance(entry, list):
            return [np.round(array.reshape(-1), decimals_) for array in entry]
        else:
            raise TypeError(f"Expected numpy array or list, got {type(entry)} instead.")

    with open(cddd_paths, 'r') as file:
        paths = json.load(file)

    command = (f"{paths['python']} {paths['wrapper']} --input {paths['input']} --output {paths['output']} "
               f"--smiles_col {smiles_col} --descriptor_col {descriptor_col} --n_cpu {n_cpus} --model_dir {paths['model']}")

    if is_polars := isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # pack the lists to strings so that they don't get broken;
    df.loc[:, smiles_col] = df[smiles_col].apply(lambda entry: ' : '.join(entry) if isinstance(entry, list) else entry)
    df.to_csv(paths['input'], sep='\t', header=True, index=False)

    os.system(command)

    df = pickle.load(open(paths['output'], 'rb'))
    df.loc[:, descriptor_col] = df[descriptor_col].apply(postprocess, decimals_=decimals)

    if is_polars:
        df = pl.from_pandas(df)

    return df


def smiles_2_mapc(smiles: Union[str, List[str]], radius: int = 2, nbits: int = 1024):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    radius: int, optional
        The radius parameter for MAPC calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """
    try:
        from mapchiral.mapchiral import encode
    except ImportError:
        raise ImportError("Function < smiles_2_mapc > requires < mapchiral > library. Please install it "
                          "with < pip install mapchiral >")

    def get_embedding(smi: str):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = encode(mol, max_radius=radius, n_permutations=nbits)
            return fp

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_mapc(df: Union[pd.DataFrame, pl.DataFrame], smiles_col: str = 'SMILES', descriptor_col: str = 'MAPC',
    radius: int = 2, nbits: int = 1024):

    try:
        from mapchiral.mapchiral import encode
    except ImportError:
        raise ImportError("Function < dataframe_2_mapc > requires < mapchiral > library. Please install it "
                          "with < pip install mapchiral >")

    if isinstance(df, pd.DataFrame):
        df[descriptor_col] = smiles_2_mapc(df[smiles_col].tolist(), radius=radius, nbits=nbits)
        return df
    elif isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series(descriptor_col, smiles_2_mapc(df[smiles_col].to_list(), radius=radius, nbits=nbits)))
        return df
    else:
        raise TypeError(f"Expected df to be either polars or pandas DataFrame, got {type(df)} instead.")
