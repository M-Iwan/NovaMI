""" Wrapper for Mordred.py due to requirement of specific python environment """


import argparse
import pandas as pd
import numpy as np
import pickle
from mordred import Calculator, descriptors
from mordred.error import Error
from rdkit import Chem
from typing import Union, List

parser = argparse.ArgumentParser(prog='Mordred', description='Calculate Mordred descriptors from SMILES')

parser.add_argument('-i', '--input', help='Path to an input file. Must be a .csv file.')
parser.add_argument('-o', '--output', help='Path to an output file. Must be a .pkl file.')
parser.add_argument('-s', '--smiles_col', help='Name of the column holding the SMILES')
parser.add_argument('-d', '--descriptor_col', help='Name of the column to which write the descriptors')

parsed_args = parser.parse_args()


def main(args):

    input_ = args.input  # .tsv file
    output_ = args.output   # .pkl file
    smiles_col_ = args.smiles_col
    descriptor_col_ = args.descriptor_col

    calc = Calculator(descriptors)

    df = pd.read_csv(input_, sep='\t')
    # unpack lists
    df.loc[:, smiles_col_] = df[smiles_col_].apply(lambda entry: entry.split(' : ') if ' : ' in entry else entry)

    def process(smiles: Union[str, List[str]], calculator_):

        if isinstance(smiles, str):
            try:
                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    print(f'Unable to construct a valid molecule from < {smiles} >')
                    return np.nan

                fp = np.array([float(value) if not isinstance(value, Error) else np.nan for value in calculator_(mol)])
                return fp.reshape(-1)

            except Exception as e:
                print(f'Could not process < {smiles} > due to the following error:\n{e}')
                return np.nan

        if isinstance(smiles, list) & all([isinstance(smi, str) for smi in smiles]):
            try:
                mols = [Chem.MolFromSmiles(smi) for smi in smiles]
                if any([mol is None for mol in mols]):
                    print(f'Unable to construct a valid molecule from < {smiles} >')
                    return np.nan

                fps = [np.array([float(value) if not isinstance(value, Error) else np.nan for value in calculator_(mol)]) for mol in mols]
                return fps

            except Exception as e:
                print(f'Unable to process < {smiles} > due to: \n{e}')
                return np.nan

        print(f'Expected < smiles > argument to be of type str, received < {type(smiles)} > instead.')
        return np.nan

    df[descriptor_col_] = df[smiles_col_].apply(process, calculator_=calc)

    pickle.dump(df, open(output_, 'wb'), protocol=4)


if __name__ == '__main__':
    main(parsed_args)
