"""
version = 0.4.0
date = 25.03.2024
author = M-Iwan
"""

import argparse
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover, RemoveStereochemistry
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt, CalcMolDescriptors, HeavyAtomCount
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

parser = argparse.ArgumentParser(prog='MolecularStandardizer',
                                 description='Uses RDKit functions to standardize SMILES')

parser.add_argument('-i', '--input', help='Path to an input file')
parser.add_argument('-o', '--output', help='Path to an output file')
parser.add_argument('-s', '--smiles_col', help="Name of the column holding the SMILES of molecules")
parser.add_argument('-t', '--target_col', help="Name of the column holding target values")
parser.add_argument('-d', '--duplicates', default=False, help="Option to autoprocess duplicates")

parsed_args = parser.parse_args()


def main(args):
    """
    Standardizes molecular SMILES using RDKit functions.
    Accepted inputs and outputs are: .xlsx, .csv, .parquet, .pkl.

    Example of usage:

    python3 /home/miwan/MolStandardizer.py --input data.csv --output data_out.csv --smiles_col SMILES

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from command line.
    """

    def validate_smiles(df: pd.DataFrame, smiles_column: str = 'SMILES'):
        """
        Use RDKit to check if a SMILES can be converted to a valid molecule and check it for potential errors.

        Parameters
        ---------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Column in df holding SMILES of molecules
        """

        RDLogger.DisableLog('rdApp.*')

        def rdkit_validate(smiles: str):

            if isinstance(smiles, str):
                mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=True)
                if mol is not None:
                    return True, 'None'
                else:
                    mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None:
                        rdkit_problems = [str(problem.GetType()) for problem in rdkit.Chem.DetectChemistryProblems(mol)]
                        return False, str(rdkit_problems)
                    else:
                        return False, 'Invalid molecule'
            else:
                return False, 'Not a string'

        out = df[smiles_column].apply(rdkit_validate).tolist()

        df.loc[:, 'Correct'] = [x[0] for x in out]
        df.loc[:, 'Issues'] = [x[1] for x in out]

        return df

    def remove_minor(df: pd.DataFrame, smiles_column: str = 'SMILES'):
        """
        Remove small fragments from SMILES, keeping only the biggest one

        Parameters
        --------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Name of a column holding SMILES
        """
        RDLogger.DisableLog('rdApp.*')

        fragment_remover = rdMolStandardize.LargestFragmentChooser()

        def process_smiles(smiles, remover):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol = remover.choose(mol)
                        return Chem.MolToSmiles(mol)
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return np.nan
                except Exception as e:
                    print(f'Could not find the biggest fragment in <{smiles}> due to {e}')
                    return np.nan
            else:
                print(f'Expected SMILES to be str, got {type(smiles)}')
                return np.nan

        df.loc[:, 'Fragment'] = df[smiles_column].apply(process_smiles, remover=fragment_remover)

        return df

    def remove_metals(df: pd.DataFrame, smiles_column: str = 'SMILES') -> pd.DataFrame:
        """
        Remove metals from molecules

        Parameters
        --------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Name of a column holding SMILES
        """
        RDLogger.DisableLog('rdApp.*')

        metal_remover = rdMolStandardize.MetalDisconnector()

        def process_smiles(smiles, remover):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol = remover.Disconnect(mol)
                        return Chem.MolToSmiles(mol)
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return np.nan
                except Exception as e:
                    print(f'Could not remove metals from <{smiles}> due to {e}')
                    return np.nan
            else:
                print(f'Expected SMILES to be str, got {type(smiles)}')
                return np.nan

        df.loc[:, 'Metal'] = df[smiles_column].apply(process_smiles, remover=metal_remover)

        return df

    def remove_salts(df: pd.DataFrame, smiles_column: str = 'SMILES'):
        """
        Remove salts from molecules

        Parameters
        --------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Name of a column holding SMILES
        """
        RDLogger.DisableLog('rdApp.*')

        salt_remover = SaltRemover.SaltRemover()

        def process_smiles(smiles, remover):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        new_smiles, deleted_list = remover.StripMolWithDeleted(mol)
                        return Chem.MolToSmiles(new_smiles), [Chem.MolToSmiles(deleted) for deleted in deleted_list]
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return np.nan, np.nan
                except Exception as e:
                    print(f'Could not remove salts <{smiles}> due to {e}')
                    return np.nan, np.nan
            else:
                print(f'Expected SMILES to be str, got {type(smiles)}')
                return np.nan, np.nan

        out = df[smiles_column].apply(process_smiles, remover=salt_remover)

        df.loc[:, 'Salt'] = [result[0] for result in out]
        df.loc[:, 'Removed'] = [result[1] for result in out]

        return df

    def remove_stereochemistry(df: pd.DataFrame, smiles_column: str = 'SMILES'):
        """
        Remove stereochemistry from molecules

        Parameters
        --------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Name of a column holding SMILES
        """

        RDLogger.DisableLog('rdApp.*')

        def process_smiles(smiles):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        RemoveStereochemistry(mol)
                        return Chem.MolToSmiles(mol)
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return np.nan
                except Exception as e:
                    print(f'Could not remove stereochemistry from <{smiles}> due to {e}')
                    return np.nan
            else:
                print(f'Expected SMILES to be str, got {type(smiles)}')
                return np.nan

        df.loc[:, 'Stereo'] = df[smiles_column].apply(process_smiles)

        return df

    def remove_charges(df: pd.DataFrame, smiles_column: str = 'SMILES'):
        """
        Attempt to neutralize charges from molecules

        Parameters
        --------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Name of a column holding SMILES
        """

        RDLogger.DisableLog('rdApp.*')

        charge_remover = rdMolStandardize.Uncharger()

        def process_smiles(smiles, remover):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol = remover.uncharge(mol)
                        return Chem.MolToSmiles(mol)
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return np.nan
                except Exception as e:
                    print(f'Could not remove charges from <{smiles}> due to {e}')
                    return np.nan
            else:
                print(f'Expected SMILES to be str, got {type(smiles)}')
                return np.nan

        df.loc[:, 'Charge'] = df[smiles_column].apply(process_smiles, remover=charge_remover)

        return df

    def remove_tautomers(df: pd.DataFrame, smiles_column: str = 'SMILES', max_num_tautomers: int = 10):
        """
        Attempt to return canonical tautomer from a molecule

        Parameters
        --------------
        df                : pd.DataFrame
                            pd.DataFrame with SMILES
        smiles_column     : str
                            Name of a column holding SMILE
        max_num_tautomers : int
                            Number of tautomers to create while evaluating the canonical
        """

        RDLogger.DisableLog('rdApp.*')

        tautomer_remover = rdMolStandardize.TautomerEnumerator()
        tautomer_remover.SetMaxTautomers(max_num_tautomers)

        def process_smiles(smiles, remover):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        mol = remover.Canonicalize(mol)
                        return Chem.MolToSmiles(mol)
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return np.nan
                except Exception as e:
                    print(f'Could not find canonical tautomer from <{smiles}> due to {e}')
                    return np.nan
            else:
                print(f'Expected SMILES to be str, got {type(smiles)}')
                return np.nan

        df.loc[:, 'Tautomer'] = df[smiles_column].apply(process_smiles, remover=tautomer_remover)

        return df

    def remove_inorganic(df: pd.DataFrame, smiles_column: str = 'SMILES'):

        """
        Remove SMILES containing inorganic parts based on SMARTS pattern matching.

        Parameters
        --------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Name of a column holding SMILES
        """

        RDLogger.DisableLog('rdApp.*')

        patterns = ['[!#1;!#6;!#7;!#8;!#9;!#17;!#35;!#53;!#15;!#16;!#3;!#5;!#11;!#12;!#19;!#20]',
                    # check for elements other than: H, Li, B, C, N, O, F, Na, Mg, P, S, Cl, K, Ca, Br, I
                    '[#6]'
                    # checks if a molecule contains at least one carbon atom
                    ]
        patterns_molecules = [Chem.MolFromSmarts(pattern) for pattern in patterns]

        def process_smiles(smiles, patterns_mol):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        if mol.HasSubstructMatch(patterns_mol[0]):  # check if a molecule has not allowed atoms
                            return True
                        elif not mol.HasSubstructMatch(patterns_mol[1]):  # check if a molecule has carbon
                            return True
                        else:
                            return False
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return False  # invalid molecule
                except Exception as e:
                    print(f'Could not convert <{smiles}> due to {e}')
                    return False  # invalid molecule
            else:
                print(f'Expected smiles to be str, got {type(smiles)}')
                return False

        df.loc[:, 'Inorganic'] = df[smiles_column].apply(process_smiles, patterns_mol=patterns_molecules)

        organic_df = df[~df['Inorganic']]
        inorganic_df = df[df['Inorganic']]

        return organic_df, inorganic_df

    def remove_duplicates(df, smiles_column, use_fingerprints: bool = False):
        """
        Function for removing duplicates based on canonical SMILES and physchem. properties

        Parameters
        --------------
        df                : pd.DataFrame
                            pd.DataFrame with SMILES
        smiles_column     : str
                            Name of a column holding SMILES
        use_fingerprints  : bool
                            An option to, aside from SMILES, also include fingerprints during detection of duplicates.
        """

        RDLogger.DisableLog('rdApp.*')

        def process_smiles(smiles):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                        fp = [value for value in np.where(np.array(fp) != 0)[0]]
                        physchem = CalcMolDescriptors(mol)
                        physchem = [np.round(value, 3) for value in physchem.values()]
                        maccs = GetMACCSKeysFingerprint(mol)
                        maccs = [value for value in np.where(np.array(maccs) != 0)[0]]
                        return fp + physchem + maccs
                    else:
                        print(f'Could not convert <{smiles}> to a valid molecule')
                        return np.nan
                except Exception as e:
                    print(f'Could not find canonical tautomer from <{smiles}> due to {e}')
                    return np.nan
            else:
                print(f'Expected SMILES to be str, got {type(smiles)}')
                return np.nan

        df.loc[:, 'Duplicated_SMILES'] = df.duplicated(subset=smiles_column, keep=False)

        if use_fingerprints is True:

            df.loc[:, 'Fingerprint'] = df[smiles_column].apply(process_smiles)
            df.loc[:, 'Duplicated_fingerprint'] = df.duplicated(subset='Fingerprint', keep=False)

            df.loc[:, 'Duplicate'] = df[['Duplicated_SMILES', 'Duplicated_fingerprint']].any(axis='columns')

            df_not_duplicates = df[~df['Duplicate']].drop(columns=['Duplicated_SMILES', 'Duplicated_fingerprint', 'Duplicate', 'Fingerprint'])
            df_duplicates = df[df['Duplicate']].sort_values(by=smiles_column)

        else:

            df_not_duplicates = df[~df['Duplicated_SMILES']].drop(columns='Duplicated_SMILES')
            df_duplicates = df[df['Duplicated_SMILES']].sort_values(by=smiles_column)

        return df_not_duplicates, df_duplicates

    def process_duplicates(df, identity_column, target_column, std_factor: float = 0.7):
        """
        Process duplicate SMILES in a DataFrame by removing outliers and updating target values.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame containing the data.
        smiles_column : str
            Name of the column containing SMILES strings.
        target_column : str
            Name of the column containing target values.
        std_factor : float, optional
            Factor used to determine the threshold for outliers. Default is 0.7.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with outliers removed and target values updated for each group of duplicate SMILES.

        Recommended values for std_factor: <1
        """
        medians = df.groupby(identity_column)[target_column].median()
        std_devs = df.groupby(identity_column)[target_column].std()

        for smiles, median, std_dev in zip(medians.index, medians, std_devs):
            low_threshold = median - std_factor * std_dev
            upp_threshold = median + std_factor * std_dev
            outliers = df[(df[identity_column] == smiles) & ((df[target_column] < low_threshold) | (df[target_column] > upp_threshold))].index
            df = df.drop(outliers)

        grouped = df.groupby(identity_column)
        for smile, group in grouped:
            mean_value = group[target_column].mean()
            first_row_index = group.head(1).index
            df.loc[first_row_index, target_column] = mean_value
            other_indices = group.index.difference(first_row_index)
            df = df.drop(other_indices)

        return df

    def filter_smiles(df: pd.DataFrame, smiles_column: str = 'SMILES'):
        """
        Remove SMILES not adhering to predefined rules

        Parameters
        --------------
        df            : pd.DataFrame
                        pd.DataFrame with SMILES
        smiles_column : str
                        Name of a column holding SMILES
        """

        RDLogger.DisableLog('rdApp.*')

        def process_smiles(smiles):
            if isinstance(smiles, str):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        logp = MolLogP(mol)
                        mol_wt = MolWt(mol)
                        num_heavy = HeavyAtomCount(mol)
                        if all([logp >= -5, logp <= 7, mol_wt >= 30, mol_wt <= 800, num_heavy >= 3, num_heavy <= 60]):
                            return True
                        else:
                            return False
                    else:
                        return False
                except Exception as e:
                    print(f'Could not convert <{smiles}> due to {e}')
                    return False
            else:
                print(f'Expected smiles to be str, got {type(smiles)}')
                return False

        df.loc[:, 'Filter'] = df[smiles_column].apply(process_smiles)

        passed = df[df['Filter']].drop(columns='Filter')
        not_passed = df[~df['Filter']].drop(columns='Filter')

        return passed, not_passed

    def read_dataframe(path, ext):
        read_path_ = f'{path}.{ext}'
        if ext == 'xlsx':
            df = pd.read_excel(read_path_)
        elif ext == 'csv':
            df = pd.read_csv(read_path_)
        elif ext == 'parquet':
            df = pd.read_parquet(read_path_)
        elif ext == 'pkl':
            df = pd.read_pickle(read_path_)
        else:
            raise ValueError('Incorrect extension')

        return df

    def write_dataframe(df, path, ext, name):
        save_path = f'{path}_{name}.{ext}'
        if len(df) >= 1:
            if ext == 'xlsx':
                df.to_excel(save_path, index=False)
            elif ext == 'csv':
                df.to_csv(save_path, index=False, header=True)
            elif ext == 'parquet':
                df.to_parquet(save_path)
            elif ext == 'pkl':
                df.to_pickle(save_path)
            else:
                raise ValueError('Incorrect extension')

    # Prepare variables
    input_ = args.input
    output_ = args.output
    smiles_column_ = str(args.smiles_col)
    target_column_ = str(args.target_col)
    clean_duplicates_ = bool(int(args.duplicates))

    in_ext = input_.split('.')[-1]
    out_ext = output_.split('.')[-1]

    read_path = input_.rstrip(f'{in_ext}').rstrip('.')
    write_path = output_.rstrip(f'{out_ext}').rstrip('.')

    print(f'--- < Now processing: {input_} > ---')

    # Read initial data
    print('< Reading data >')
    data = read_dataframe(read_path, in_ext)

    # Add Mol_ID and validate initial smiles
    print('< Adding Mol_ID >')
    print('< Performing initial validation of SMILES >')
    data.loc[:, 'Mol_ID'] = np.arange(0, data.shape[0])
    data = validate_smiles(data, smiles_column_)

    write_dataframe(data[~data['Correct']], write_path, out_ext, 'invalid_smiles')
    data = data[data['Correct']].drop(columns=['Correct', 'Issues'])

    # Disconnect metals
    print('< Disconnecting metals >')
    data = remove_metals(data, smiles_column_)
    data = validate_smiles(data, 'Metal')

    write_dataframe(data[~data['Correct']], write_path, out_ext, 'invalid_metal')
    data = data[data['Correct']].drop(columns=['Correct', 'Issues'])

    # Remove salts
    print('< Removing common salts >')
    data = remove_salts(data, 'Metal')
    data = validate_smiles(data, 'Salt')

    write_dataframe(data[~data['Correct']], write_path, out_ext, 'invalid_salt')
    data = data[data['Correct']].drop(columns=['Correct', 'Issues', 'Metal'])

    # Remove compounds with inorganic parts
    print('< Removing inorganic compounds >')
    data, inorganic = remove_inorganic(data, 'Salt')
    write_dataframe(inorganic, write_path, out_ext, 'inorganic')
    data = data.drop(columns='Inorganic')

    # Select biggest fragments
    print('< Selecting major fragments >')
    data = remove_minor(data, 'Salt')
    data = validate_smiles(data, 'Fragment')

    write_dataframe(data[~data['Correct']], write_path, out_ext, 'invalid_fragment')
    data = data[data['Correct']].drop(columns=['Correct', 'Issues', 'Salt', 'Removed'])

    # Neutralize charges
    print('< Neutralizing charges >')
    data = remove_charges(data, 'Fragment')
    data = validate_smiles(data, 'Charge')

    write_dataframe(data[~data['Correct']], write_path, out_ext, 'invalid_charge')
    data = data[data['Correct']].drop(columns=['Correct', 'Issues', 'Fragment'])

    # Remove stereochemistry
    print('< Removing stereochemistry >')
    data = remove_stereochemistry(data, 'Charge')
    data = validate_smiles(data, 'Stereo')

    write_dataframe(data[~data['Correct']], write_path, out_ext, 'invalid_stereochemistry')
    data = data[data['Correct']].drop(columns=['Correct', 'Issues', 'Charge'])

    # Select canonical tautomer
    print('< Searching for a canonical tautomer >')
    data = remove_tautomers(data, 'Stereo', 50)
    data = validate_smiles(data, 'Tautomer')

    write_dataframe(data[~data['Correct']], write_path, out_ext, 'invalid_tautomer')
    data = data[data['Correct']].drop(columns=['Correct', 'Issues', 'Stereo'])

    # Canonicalize SMILES
    print('< Making canonical SMILES >')
    data.loc[:, 'SMILES_final'] = data['Tautomer'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    data = data.drop(columns='Tautomer')

    # Remove duplicates
    print('< Detecting duplicated entries >')
    data, duplicates = remove_duplicates(data, 'SMILES_final', use_fingerprints=False)

    write_dataframe(duplicates, write_path, out_ext, 'duplicates')

    if clean_duplicates_ is True:
        print('< Processing duplicated entries >')
        duplicates = process_duplicates(duplicates, identity_column='SMILES_final', target_column=target_column_, std_factor=0.7)
        write_dataframe(duplicates, write_path, out_ext, 'processed_duplicates')

    # Filter compounds
    print('< Filtering compounds >')
    data, failed = filter_smiles(data, 'SMILES_final')

    data = data.sort_values(by='Mol_ID', ascending=True)
    write_dataframe(failed, write_path, out_ext, 'filtered')
    write_dataframe(data, write_path, out_ext, 'final')

    print('< Finished successfully >')


if __name__ == '__main__':
    main(parsed_args)
