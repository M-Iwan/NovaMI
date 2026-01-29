import argparse
import numpy as np
import pandas as pd
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import time
import rdkit
from rdkit.Chem import AllChem
from rdkit import RDLogger

parser = argparse.ArgumentParser(prog='ChemicalIdentityResolver',
                                 description='Uses NCI/NIH CIR to find SMILES for a given name')

parser.add_argument('-i', '--input', help='Absolute path to an input file')
parser.add_argument('-o', '--output', help='Absolute path to an output file')
parser.add_argument('-n', '--name', help="Name of the column holding molecule's names")
parser.add_argument('-r', '--resolver', help='Type of resolved identity')

parsed_args = parser.parse_args()


def main(args):
    """
    TODO: add modules for using common chemistry CAS resolver
    TODO: add modules for using ECHA website for resolving names/CAS
    Resolves chemical identities representations using the NCI/NIH CIR API.
    Accepted inputs and outputs are: .xlsx, .csv, .parquet, .pkl.
    Accepted keywords for resolver are: smiles, stdinchi, stdinchikey, cas, iupac_name

    Example of usage:

    python3 CIR.py --input data.csv --output data_out.csv --name molecules --resolver smiles --strip_salts True

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from command line.
    """

    def chemical_resolver(name: str, resolver: str = 'smiles', sleep_time: int = 6, max_retries: int = 5):
        """
        Resolve chemical identity a single molecule using NCI/NIH CIR API

        Parameters
        ---------------
        name   : str
                 name of a molecule
        resolver : str
                 type of output. available are: smiles, stdinchi, stdinchikey, cas, iupac_name
        sleep_time : int
                     time to wait between attempting a retry
        max_retries : int
                      maximum number of attempts to make
        """

        name = name.lower().strip().replace(' ', '%20')
        url = f'https://cactus.nci.nih.gov/chemical/structure/{name.lower()}/{resolver.lower()}'

        try:
            out = urlopen(url, data=None, timeout=8).read().decode()
            return out
        except HTTPError as e:
            print(f'Failed to resolve "{name.replace("%20", " ")}" due to {e}')
            try:
                out = urlopen(url+'?resolver=name_by_chemspider', data=None, timeout=16).read().decode()
                return out
            except HTTPError as e:
                print(f'Failed to resolve "{name.replace("%20", " ")}" due to {e}')
                return np.nan
            except Exception:
                return np.nan
        except URLError or ConnectionResetError as e:
            print(f'Connection lost while resolving "{name.replace("%20", " ")}" due to {e}. '
                  f'Retrying in {sleep_time} seconds.')
            retries = 1
            while retries <= max_retries:
                try:
                    time.sleep(sleep_time)
                    sleep_time += 2
                    out = urlopen(url, data=None).read().decode()
                    print(f'"{name.replace("%20", " ")}" solved successfully')
                    return out
                except HTTPError as e:
                    print(f'Failed to resolve "{name.replace("%20", " ")}" after {retries} attempts due to {e}')
                    return np.nan
                except URLError or ConnectionResetError as e:
                    print(f'Connection lost while resolving "{name.replace("%20", " ")}" due to {e}. '
                          f'Retrying in {sleep_time} seconds.')
                    pass
                retries += 1
            print(f'Failed to resolve "{name.replace("%20", " ")}" after {max_retries} attempts')
            return np.nan
        except UnicodeEncodeError:
            print(f'Could not properly encode {name}. Skipping.')
            return np.nan

    def validate_smiles(df: pd.DataFrame, smiles_col: str = 'SMILES'):

        RDLogger.DisableLog('rdApp.*')

        """
        Use RDKit to check if a SMILES can be converted to a valid molecule and check it for potential errors.

        Parameters
        ---------------
        df : pd.DataFrame
             with SMILES
        smiles_col : str
                     Column in df holding SMILES of molecules
        """

        def rdkit_validate(smiles: str):

            try:
                mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=True)
                if mol is not None:
                    return True, None
                else:
                    mol = rdkit.Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None:
                        rdkit_problems = [problem.GetType() for problem in rdkit.Chem.DetectChemistryProblems(mol)]
                        return False, rdkit_problems
                    else:
                        return False, 'Invalid molecule'
            except Exception as e:
                return False, 'Invalid molecule'

        out = df[smiles_col].apply(rdkit_validate).tolist()

        df['Correct'] = [x[0] for x in out]
        df['Issues'] = [x[1] for x in out]

        return df

    input_ = args.input
    output_ = args.output
    name_ = args.name
    resolver_ = args.resolver

    if resolver_ not in ['smiles', 'stdinchi', 'stdinchikey', 'cas', 'iupac_name']:
        raise ValueError('Wrong format passed. Expected one of: smiles, stdinchi, stdinchikey, cas, iupac_name')

    in_ext = input_.split('.')[-1]
    out_ext = output_.split('.')[-1]

    if in_ext == 'xlsx':
        data = pd.read_excel(input_)
    elif in_ext == 'csv':
        data = pd.read_csv(input_)
    elif in_ext == 'parquet':
        data = pd.read_parquet(input_)
    elif in_ext == 'pkl':
        data = pd.read_pickle(input_)
    else:
        raise ValueError(f'Unrecognized file extension. Expected one of: .xlsx, .csv, .parquet, .pkl')

    data['Mol_ID'] = [x for x in range(len(data))]
    data['Output'] = data[name_].apply(chemical_resolver, resolver=resolver_)

    solved = data.dropna(subset='Output', axis=0)
    unsolved = data.drop(index=solved.index)

    write_path = output_.rstrip(f'.{out_ext}')

    if str(resolver_) == 'smiles':
        solved = validate_smiles(solved, 'Output')

    if out_ext == 'xlsx':
        solved.to_excel(f'{write_path}_resolved.xlsx', index=False)
        unsolved.to_excel(f'{write_path}_unsolved.xlsx', index=False)
    elif out_ext == 'csv':
        solved.to_csv(f'{write_path}_resolved.csv', header=True, index=False)
        unsolved.to_csv(f'{write_path}_unsolved.csv', header=True, index=False)
    elif out_ext == 'parquet':
        solved.to_parquet(f'{write_path}_resolved.parquet')
        unsolved.to_parquet(f'{write_path}_unsolved.parquet')
    elif out_ext == 'pkl':
        solved.to_pickle(f'{write_path}_resolved.pkl')
        unsolved.to_pickle(f'{write_path}_unsolved.pkl')
    else:
        raise ValueError(f'Unrecognized file extension. Expected one of: .xlsx, .csv, .parquet, .pkl')


if __name__ == '__main__':
    main(parsed_args)
