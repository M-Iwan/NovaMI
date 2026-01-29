import pickle
import joblib
from typing import Optional, Dict

import pandas as pd
import polars as pl


def read_pd(path: str, kwargs: Optional[Dict] = None) -> pd.DataFrame:
    if kwargs is None:
        kwargs = {}
    ext = path.split('.')[-1]

    if ext == 'xlsx':
        df = pd.read_excel(path, **kwargs)
    elif ext == 'csv':
        df = pd.read_csv(path, **kwargs)
    elif ext == 'tsv':
        df = pd.read_csv(path, sep='\t', **kwargs)
    elif ext == 'parquet':
        df = pd.read_parquet(path, **kwargs)
    elif ext == 'pkl':
        df = pickle.load(open(path, 'rb'))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected the file to be pandas DataFrame, got {type(df)} instead.")
    else:
        raise ValueError('Incorrect extension')

    return df


def write_pd(df, path):
    ext = path.split('.')[-1]
    if len(df) >= 1:
        if ext == 'xlsx':
            df.to_excel(path, index=False)
        elif ext == 'csv':
            df.to_csv(path, index=False, header=True)
        elif ext == 'tsv':
            df.to_csv(path, index=False, header=True, sep='\t')
        elif ext == 'parquet':
            df.to_parquet(path)
        elif ext == 'pkl':
            df.to_pickle(path)
        else:
            raise ValueError('Incorrect extension')


def read_pl(path: str, kwargs: Optional[Dict] = None) -> pl.DataFrame:
    """
    Read DataFrame-like structure using the Polars library
    """
    if kwargs is None:
        kwargs = {}

    ext = path.split('.')[-1]
    if ext == 'xlsx':
        df = pl.read_excel(path, **kwargs)
    elif ext == 'csv':
        df = pl.read_csv(path, **kwargs)
    elif ext == 'tsv':
        df = pl.read_csv(path, separator='\t', **kwargs)
    elif ext == 'parquet':
        df = pl.read_parquet(path, **kwargs)
    elif ext == 'pkl':
        df = pickle.load(open(path, 'rb'))
    elif ext == 'joblib':
        df = joblib.load(path)
    else:
        raise ValueError(f'Extension .{ext} not supported.')

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    return df


def write_pl(df: pl.DataFrame, path: str, kwargs: Optional[Dict] = None) -> None:
    if kwargs is None:
        kwargs = {}
    ext = path.split('.')[-1]
    if len(df) >= 1:
        if ext == 'xlsx':
            df.write_excel(path, **kwargs)
        elif ext == 'csv':
            df.write_csv(path, **kwargs)
        elif ext == 'tsv':
            df.write_csv(path, separator='\t', **kwargs)
        elif ext == 'parquet':
            df.write_parquet(path, **kwargs)
        elif ext == 'pkl':
            pickle.dump(df, open(path, 'wb'))
        elif ext == 'joblib':
            joblib.dump(df, path)
        else:
            raise ValueError(f'Extension .{ext} not supported.')
    else:
        print('Empty DataFrame')
            