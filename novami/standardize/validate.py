import pandas as pd
import rdkit
from rdkit import RDLogger


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

        if not isinstance(smiles, str):
            return False, 'Not a string'

        try:
            mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is not None:
                return True, 'None'
            else:
                mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is None:
                    return False, 'Invalid SMILES'

                rdkit_problems = [str(problem.GetType()) for problem in rdkit.Chem.DetectChemistryProblems(mol)]
                return False, str(rdkit_problems)
        except Exception as e:
            print(f'Encountered unexpected error: {e}')
            return False, 'Unknown error'

    out = df[smiles_column].apply(rdkit_validate).tolist()

    df.loc[:, 'Correct'] = [x[0] for x in out]
    df.loc[:, 'Issues'] = [x[1] for x in out]

    return df