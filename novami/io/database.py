from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from novami.standardize.duplicates import process_duplicates


def preprocess_binding_db(file_path: str) -> pd.DataFrame:
    """
    Preprocess a BindingDB SDF file to create a standardized DataFrame of compounds with IC50 values.

    Parameters
    ----------
    file_path : str
        Path to the BindingDB SDF file to be processed.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame containing molecular information and standardized IC50 values.
        The DataFrame includes the following columns:
        - InChI: InChI representation of the molecule
        - InChI_key: InChI key identifier
        - Protein: Target protein name
        - ChEMBL ID: ChEMBL identifier for the compound
        - PDB_ID: Associated PDB identifiers if available
        - SMILES: SMILES representation of the molecule
        - Channel: Ion channel identifier derived from the filename
        - Value: IC50 value in μM
        - Unit: Concentration unit (μM)
        - Relation: Relationship symbol (=, >, <)
        - pIC50: -log10 of molar IC50

    Notes
    -----
    The preprocessing workflow includes:
    1. Loading the SDF file and selecting relevant columns
    2. Converting molecular structures to SMILES format
    3. Standardizing IC50 units to μM and calculating pIC50
    4. Filtering for entries with exact values (not inequalities)
    5. Identifying and handling duplicate entries:
    """

    def check_duplicates(value: float, other_values: List[float]): # based on pChEMBL_Value
        if value in other_values:
            return 'Duplicate'
        if any([value + 3 in other_values, value - 3 in other_values, value + 6 in other_values, value - 6 in other_values]):
            return 'Unit Error'
        else:
            return 'Cool&Good'

    def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        has_duplicate = []

        for idx, row in df.iterrows():
            value = row['Value'].value
            smiles = row['SMILES'].value

            other_values = df[df['SMILES'] == smiles]['Value'].drop(idx).tolist()

            has_duplicate.append(check_duplicates(value, other_values))

        df['Issues'] = has_duplicate
        return df

    def convert_ic50(value: str):
        value = value.lower().strip()
        if value.startswith(('>', '<')):
            relation = value[0]
            value = value[1:]
        else:
            relation = '='
        try:
            value = float(value) / 1000
        except ValueError:
            print(f'Cannot convert {value} to float')
        return value, 'uM', relation

    file_name = file_path.split('/')[-1]
    df = PandasTools.LoadSDF(file_path, molColName='Mol')

    df = df[['Ligand InChI', 'Ligand InChI Key', 'Target Name', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
             'ChEMBL ID of Ligand', 'PDB ID(s) for Ligand-Target Complex', 'Mol']]

    df = df.rename(columns=
                    {'Ligand InChI': 'InChI',
                    'Ligand InChI Key': 'InChI_key',
                    'Target Name': 'Protein',
                    'ChEMBL ID of Ligand': 'ChEMBL ID',
                    'PDB ID(s) for Ligand-Target Complex': 'PDB_ID'}
                    )

    df['SMILES'] = df['Mol'].apply(Chem.MolToSmiles)

    # we don't need Mol anymore, the others are usually missing
    df = df.drop(columns=['Mol', 'Ki (nM)', 'Kd (nM)', 'EC50 (nM)'])
    df = df.replace('', pd.NA)
    df['File'] = file_name.split('.')[0].replace('_', '.')

    df = df.dropna(subset=['IC50 (nM)', 'SMILES'], how='any').reset_index(drop=True)

    # Convert IC50 to numerical and apply ChEMBL formatting; the Value is in uM
    df[['Value', 'Unit', 'Relation']] = df['IC50 (nM)'].apply(lambda value: pd.Series(convert_ic50(value)))
    df['pIC50'] = df['Value'].apply(lambda value: -np.log10(value/10**6))

    # Only use entries with known values
    df = df[df.Relation == '='].reset_index(drop=True)

    # Remove duplicated entries
    df = mark_duplicates(df)

    df = df[df['Issues'] != 'Unit Error'].reset_index(drop=True)  # remove the 3/6 differing ones

    g_df = df[df['Issues'] == 'Cool&Good']
    d_df = df[df['Issues'] == 'Duplicate']

    pd_df = d_df.groupby('SMILES')[d_df.columns].apply(process_duplicates).reset_index(drop=True)

    df = pd.concat([g_df, pd_df], ignore_index=True)
    df = df.drop(columns=['IC50 (nM)', 'Issues'])
    return df


def preprocess_chembl(df: pd.DataFrame, activity_type: str = 'IC50') -> pd.DataFrame:
    """
    Preprocess ChEMBL data to obtain standardized bioactivity measurements.

    This function filters and cleans ChEMBL data to extract high-quality bioactivity
    measurements for a specified activity type. It handles duplicate entries,
    standardizes molecular representations, and applies quality filters to ensure
    consistency and reliability of the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing ChEMBL data with columns such as 'Standard Type',
        'Smiles', 'pChEMBL Value', etc.

    activity_type : str, optional
        Type of activity to filter for. Must be one of 'IC50', 'Kd', or 'Ki'.
        Default is 'IC50'.

    Returns
    -------
    pd.DataFrame
        A cleaned and filtered DataFrame containing standardized bioactivity data.
        Returns an empty DataFrame if no entries meet the filtering criteria.

    Notes
    -----
    The preprocessing workflow includes:
    1. Removing unnecessary columns and renaming remaining ones
    2. Filtering for entries with complete information:
       - Exact measurements (relation = '=')
       - No validity concerns
       - Units in nM
       - Single protein targets
       - Available pChEMBL values
    3. Excluding measurements from protein variants or mutations
    4. Restricting to assays using human (Homo sapiens) cell lines
    5. Converting SMILES to canonical RDKit representation
    6. Handling duplicates

    The function prints progress information at each filtering step.
    """

    if activity_type not in ['IC50', 'Kd', 'Ki']:
        raise ValueError('Activity type must be in ["IC50", "Kd", "Ki"]')

    to_drop = ['Molecule Max Phase', 'Molecular Weight', '#RO5 Violations', 'AlogP', 'Ligand Efficiency BEI',
               'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 'Compound Key',
               'Source Description', 'Document Journal', 'Uo Units', 'Cell ChEMBL ID', 'Properties', 'Action Type',
               'Standard Text Value', 'Value', 'Potential Duplicate', 'Comment', 'BAO Format ID',
               'BAO Label', 'Target Organism']

    df = df.drop(columns=[col for col in to_drop if col in df.columns])

    # Renaming because THESE ARE LONG
    to_rename = {'Molecule ChEMBL ID': 'ID_Mol', 'Molecule Name': 'Name', 'Smiles': 'SMILES',
                 'Standard Relation': 'Relation', 'Standard Type': 'Type', 'Standard Value': 'Value',
                 'Standard Units': 'Units', 'Assay ChEMBL ID': 'ID_Assay', 'Target ChEMBL ID': 'ID_Target',
                 'pChEMBL Value': 'pChEMBL_Value', 'Assay Description': 'Assay_Desc', 'Assay Type': 'Assay_Type',
                 'Target Name': 'Target', 'Document Year': 'Year', 'Data Validity Comment': 'Validity',
                 'Target Type': 'Type_Target', 'Document ChEMBL ID': 'Document', 'Source ID': 'ID_Source',
                 'Assay Variant Accession': 'Assay_Variant_Accession', 'Assay Variant Mutation': 'Assay_Variant_Mutation',
                 'Assay Tissue ChEMBL ID': 'Assay_Tissue_ID', 'Assay Tissue Name': 'Assay_Tissue_Name',
                 'Assay Subcellular Fraction': 'Assay_Subcellular', 'Assay Parameters': 'Assay_Parameters',
                 'Assay Organism': 'Assay_Organism', 'Assay Cell Type': 'Assay_Cell_Type'}

    def check_duplicates(value: float, other_values: List[float]): # based on pChEMBL_Value
        if value in other_values:
            return 'Duplicate'
        if any([value + 3 in other_values, value - 3 in other_values, value + 6 in other_values, value - 6 in other_values]):
            return 'Unit Error'
        else:
            return 'Cool&Good'

    def mark_duplicates(df):
        has_duplicate = []

        for idx, row in df.iterrows():
            value = row['pChEMBL_Value']
            smiles = row['SMILES']

            other_values = df[df['SMILES'] == smiles]['pChEMBL_Value'].drop(idx).values

            has_duplicate.append(check_duplicates(value, other_values))

        df['Issues'] = has_duplicate
        return df

    num_entries = len(df)

    df = df.rename(columns={key: value for key, value in to_rename.items() if key in df.columns})
    df = df.replace(np.nan, pd.NA)

    to_type = {'ID_Mol': 'string', 'Name': 'string', 'SMILES': 'string', 'Type': 'string', 'Relation': 'string',
               'Value': 'Float32', 'Units': 'string', 'pChEMBL_Value': 'Float32', 'ID_Assay': 'string',
               'Assay_Desc': 'string', 'Assay_Type': 'string', 'ID_Target': 'string', 'Target': 'string',
               'Validity': 'string', 'Type_Target': 'string', 'Document': 'string'}

    df = df.astype({key: value for key, value in to_type.items() if key in df.columns})
    df['Relation'] = df['Relation'].apply(lambda string: string.strip("'"))

    print(f'Initial number of entries: {num_entries}')

    # Use only entries with full information available
    df = df[(df.Type == activity_type) & (df.Relation == '=') & (df.Validity.isnull()) & (df.Units == 'nM') &
            (df.Type_Target == 'SINGLE PROTEIN') & (df.pChEMBL_Value.notnull())].reset_index(drop=True)

    print(f'Checking completeness of data. Dropped {num_entries - len(df)} entries.')
    num_entries = len(df)
    if num_entries == 0:
        print('No entries remaining')
        return pd.DataFrame()
    print(f'Number of entries remaining: {num_entries}')

    # Assay shouldn't use a variant of a protein and should use Homo sapiens cell lines
    matches = ['mutant', 'mutation', 'variant']
    df['Assay_Desc'] = df['Assay_Desc'].str.lower().str.strip()
    df = df[[all([match not in row['Assay_Desc'] for match in matches]) for idx, row in df.iterrows()]].reset_index(drop=True)
    df = df[df['Assay_Organism'] == 'Homo sapiens']

    print(f'Checking for protein variants. Dropped {num_entries - len(df)} entries.')
    num_entries = len(df)
    if num_entries == 0:
        print('No entries remaining')
        return pd.DataFrame()
    print(f'Number of entries remaining: {num_entries}')

    # convert to RDKit-native representation
    df['SMILES'] = df['SMILES'].apply(lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))

    # Measurements with equal values are duplicates, differing by 3/6 pX units suggests entry errors
    df = mark_duplicates(df)
    df = df[df['Issues'] != 'Unit Error'].reset_index(drop=True)  # remove the 3/6 differing ones

    g_df = df[df['Issues'] == 'Cool&Good']
    d_df = df[df['Issues'] == 'Duplicate']

    pd_df = d_df.groupby('SMILES').apply(process_duplicates, pIC50_name='pChEMBL_Value').reset_index(drop=True)
    df = pd.concat([g_df, pd_df], ignore_index=True)

    print(f'Checking for duplicate entries. Dropped {num_entries - len(df)} entries.')
    num_entries = len(df)
    if num_entries == 0:
        print('No entries remaining')
        return pd.DataFrame()
    print(f'Number of entries remaining: {num_entries}')

    df = df.drop(columns='Issues')

    return df