import os
import json

import polars as pl


def opsin_convert(df: pl.DataFrame, opsin_paths: str, name_col: str = 'Name', out_col: str = 'SMILES'):
    """
    Convert IUPAC names to SMILES

    Parameters
    ----------

    df: pl.DataFrame
        A polars DataFrame
    opsin_paths: str
        Path to a JSON file containing paths for OPSIN-related files. Must contain path to opsin .jar file ['opsin']
        and to a directory with read/write permission ['io']
    name_col: str
        Name of the column with IUPAC names of chemicals
    out_col: str
        Name of the column for output
    """

    with open(opsin_paths, 'r') as file:
        paths = json.load(file)

    in_path = os.path.join(paths['io'], 'names_in.txt')
    out_path = os.path.join(paths['io'], 'names_out.txt')

    names = df[name_col].to_list()

    with open(in_path, 'w') as f:
        for name in names:
            f.write(name + '\n')

    open(out_path, 'w')

    command = f"java -jar {paths['opsin']} -osmi {in_path} {out_path}"
    os.system(command)

    smiles = open(out_path, 'r').readlines()

    df = df.with_columns(pl.Series(out_col, smiles))
    return df
