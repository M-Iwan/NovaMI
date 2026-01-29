import pandas as pd
from rdkit import Chem, RDLogger


def filter_inorganic(df: pd.DataFrame, smiles_column: str = 'SMILES', out_column: str = 'Inorganic'):
    """
    Remove SMILES containing inorganic parts based on SMARTS pattern matching.

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILES
    out_column : str
        Name of the output column

    Notes
    -----
    Nested function returns True for molecules that are inorganic
    """

    RDLogger.DisableLog('rdApp.*')

    patterns = ['[!#1;!#6;!#7;!#8;!#9;!#17;!#35;!#53;!#15;!#16;!#3;!#5;!#11;!#12;!#19;!#20]',
                # check for elements other than: H, Li, B, C, N, O, F, Na, Mg, P, S, Cl, K, Ca, Br, I
                '[#6]'
                # checks if a molecule contains at least one carbon atom
                ]
    patterns_molecules = [Chem.MolFromSmarts(pattern) for pattern in patterns]

    def process_smiles(smiles, patterns_mol):
        if not isinstance(smiles, str):
            print(f'Expected smiles to be of type str, got {type(smiles)} instead')
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return False  # invalid molecule

            if mol.HasSubstructMatch(patterns_mol[0]):  # check if a molecule has not allowed atoms
                return True
            elif not mol.HasSubstructMatch(patterns_mol[1]):  # check if a molecule has carbon
                return True
            else:
                return False

        except Exception as e:
            print(f'Could not process <{smiles}> due to {e}')
            return False  # invalid molecule

    df.loc[:, out_column] = df[smiles_column].apply(process_smiles, patterns_mol=patterns_molecules)

    org_df = df[~df[out_column]]
    inorg_df = df[df[out_column]]
    return org_df, inorg_df


def filter_smiles(df: pd.DataFrame, smiles_column: str = 'SMILES'):
    """
    Remove SMILES not adhering to predefined rules

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILES
    """

    RDLogger.DisableLog('rdApp.*')

    def process_smiles(smiles):
        if not isinstance(smiles, str):
            print(f'Expected smiles to be of type str, got {type(smiles)} instead')
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return False

            logp = MolLogP(mol)
            mol_wt = MolWt(mol)
            num_heavy = Descriptors.HeavyAtomCount(mol)
            if (logp >= -5) & (logp <= 7) & (mol_wt >= 30) & (mol_wt <= 800) & (num_heavy >= 3) & (num_heavy <= 60):
                return True
            else:
                return False

        except Exception as e:
            print(f'Could not process <{smiles}> due to {e}')
            return False

    df.loc[:, 'Filter'] = df[smiles_column].apply(process_smiles)

    passed = df[df['Filter']].drop(columns='Filter')
    not_passed = df[~df['Filter']].drop(columns='Filter')

    return passed, not_passed
