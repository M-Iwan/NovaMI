import polars as pl
from rdkit import Chem
from rdkit.Chem import PandasTools

from novami.standardize.duplicates import mad_duplicates, check_unit_error


def preprocess_binding_db(file_path: str, range_threshold: float = 1.0, z_threshold: float = 3.5) -> pl.DataFrame:
    """
    Preprocess a BindingDB SDF file to create a standardized DataFrame of compounds with IC50 values.

    Parameters
    ----------
    file_path : str
        Path to the BindingDB SDF file to be processed.
    range_threshold: float
        Maximum difference between values to consider them consistent. Default is 1.0
    z_threshold: float
        Maximum threshold for MAD outlier detection. Default is 3.5

    Returns
    -------
    pl.DataFrame
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
    5. Identifying and handling duplicate entries using MAD values
    """

    def convert_ic50(value: str):
        """
        The IC50 is in format [<,>][Value]
        """
        if value == "" or value is None:
            value, relation = None, None
        else:
            relation = value[0] if value[0] in ["<", ">"] else "="
            value = value[1:]
            try:
                value = float(value) / 1000
            except ValueError:
                f'Cannot convert {value} to float'
        return {
            "Value": value,
            "Unit": "uM",
            "Relation": relation
        }

    df = PandasTools.LoadSDF(file_path, molColName='Mol')
    df['SMILES'] = df['Mol'].apply(lambda mol: Chem.MolToSmiles(mol))
    df = df.drop(columns='Mol')
    df = pl.from_pandas(df)

    df = (df
        .select(['SMILES', 'Ligand InChI', 'Ligand InChI Key', 'Target Name', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
            'ChEMBL ID of Ligand', 'PDB ID(s) for Ligand-Target Complex'])
        .rename({
            'Ligand InChI': 'InChI',
            'Ligand InChI Key': 'InChI_key',
            'Target Name': 'Protein',
            'ChEMBL ID of Ligand': 'ChEMBL ID',
            'PDB ID(s) for Ligand-Target Complex': 'PDB_ID'})
        .drop(
            ['Ki (nM)', 'Kd (nM)', 'EC50 (nM)'])
    )

    df = (df
        .with_columns([
            pl.col(col_name).replace(old='', new=None) for col_name in df.columns
        ])
        .drop_nulls(
            subset=['IC50 (nM)', 'SMILES']
        )
    )

    df = df.with_columns(
        pl.col("IC50 (nM)").map_elements(
            lambda entry: convert_ic50(entry),
            return_dtype=pl.Struct({
                "Value": pl.Float64,
                "Unit": pl.String,
                "Relation": pl.String
            })
        ).alias("_expanded")
    ).unnest("_expanded")

    df = (df
        .with_columns(
            ((pl.col("Value")/10**6).log10()*(-1)).alias("pIC50")
        )
        .filter(pl.col("Relation") == "=")
    )

    # Remove normal duplicated values
    er_df = df.filter(~df.select(["SMILES", "pIC50"]).is_duplicated())

    # Remove measurements differing by 3/6/9 log10 units
    er_df = (er_df
        .group_by("SMILES")
        .agg(
            pl.col("pIC50").map_batches(
                lambda ser: check_unit_error(ser.to_list()),
                return_dtype=pl.Boolean,
                returns_scalar=True
            ).alias("UnitError")
        )
        .filter(~pl.col("UnitError"))
    )

    df = df.join(er_df, how='inner', on='SMILES').drop("UnitError")

    df = mad_duplicates(
        df=df,
        smiles_col="SMILES",
        range_threshold=range_threshold,
        z_threshold=z_threshold
    )

    df = df.filter(pl.col("pIC50").is_not_null())

    return df


def preprocess_chembl(df: pl.DataFrame, activity_type: str = 'IC50', range_threshold: float = 1.0,
                      z_threshold: float = 3.5) -> pl.DataFrame:
    """
    Preprocess ChEMBL data to obtain standardized bioactivity measurements.

    This function filters and cleans ChEMBL data to extract high-quality bioactivity
    measurements for a specified activity type. It handles duplicate entries,
    standardizes molecular representations, and applies quality filters to ensure
    consistency and reliability of the data.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing ChEMBL data
    activity_type : str, optional
        Type of activity to filter for. Must be one of 'IC50', 'Kd', or 'Ki'.
    range_threshold: float
        Maximum difference between values to consider them consistent. Default is 1.0
    z_threshold: float
        Maximum threshold for MAD outlier detection. Default is 3.5

    Returns
    -------
    pl.DataFrame
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
    5. Handling duplicates
    """

    if activity_type not in ['IC50', 'Kd', 'Ki']:
        raise ValueError('Activity type must be in ["IC50", "Kd", "Ki"]')

    to_drop = ['Molecule Max Phase', 'Molecular Weight', '#RO5 Violations', 'AlogP', 'Ligand Efficiency BEI',
               'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 'Compound Key',
               'Source Description', 'Document Journal', 'Uo Units', 'Cell ChEMBL ID', 'Properties', 'Action Type',
               'Standard Text Value', 'Value', 'Potential Duplicate', 'Comment', 'BAO Format ID', 'BAO Label',
               'Target Organism']

    df = df.drop([col for col in to_drop if col in df.columns])

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

    num_entries = len(df)

    df = df.rename({key: value for key, value in to_rename.items() if key in df.columns})
    df = df.with_columns([
        pl.col(col_name).replace("", None).alias(col_name) for col_name in df.columns if df.schema[col_name] == pl.String
    ])

    to_type = {'ID_Mol': pl.String, 'Name': pl.String, 'SMILES': pl.String, 'Type': pl.String, 'Relation': pl.String,
               'Value': pl.Float64, 'Units': pl.String, 'pChEMBL_Value': pl.Float64, 'ID_Assay': pl.String,
               'Assay_Desc': pl.String, 'Assay_Type': pl.String, 'ID_Target': pl.String, 'Target': pl.String,
               'Validity': pl.String, 'Type_Target': pl.String, 'Document': pl.String}

    df = df.cast(to_type)
    df = df.with_columns(
        pl.col("Relation").str.strip_chars("'")
    )

    print(f'Initial number of entries: {num_entries}')

    # Use only entries with full information available
    df = df.filter(
        (pl.col("Type") == activity_type) &
        (pl.col("Relation") == "=") &
        (pl.col("Validity").is_null()) &
        (pl.col("Units") == "nM") &
        (pl.col("Type_Target") == "SINGLE PROTEIN") &
        (pl.col("pChEMBL_Value").is_not_null()) &
        (pl.col("Assay_Organism") == "Homo sapiens")
    )

    print(f'Checking completeness of data. Dropped {num_entries - len(df)} entries.')
    num_entries = len(df)
    if num_entries == 0:
        print('No entries remaining')
        return pl.DataFrame()
    print(f'Number of entries remaining: {num_entries}')

    # Assay shouldn't use a variant of a protein and should use Homo sapiens cell lines
    patterns = ['mutant', 'mutation', 'variant']

    df = (df
        .with_columns(
            pl.col("Assay_Desc").str.to_lowercase().str.strip_chars().alias("Assay_Desc")
        )
        .filter(
            ~pl.col("Assay_Desc").str.contains_any(patterns)  # So nice they have this
        )
    )

    print(f'Checking for protein variants. Dropped {num_entries - len(df)} entries.')
    num_entries = len(df)
    if num_entries == 0:
        print('No entries remaining')
        return pl.DataFrame()
    print(f'Number of entries remaining: {num_entries}')

    df = df.with_columns(
        pl.col("SMILES").map_elements(
            lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles)),
            return_dtype=pl.String,
        )
    )

    # Remove exact duplicates
    er_df = df.filter(~df.select(["SMILES", "pChEMBL_Value"]).is_duplicated())

    # Remove measurements differing by 3/6/9 log10 units
    er_df = (er_df
        .group_by("SMILES")
        .agg(
            pl.col("pChEMBL_Value").map_batches(
                lambda ser: check_unit_error(ser.to_list()),
                return_dtype=pl.Boolean,
                returns_scalar=True
            ).alias("UnitError")
        )
        .filter(~pl.col("UnitError"))
    )

    df = df.join(er_df, how='inner', on='SMILES').drop("UnitError")

    df = mad_duplicates(
        df=df,
        smiles_col="SMILES",
        value_col="pChEMBL_Value",
        range_threshold=range_threshold,
        z_threshold=z_threshold
    )

    print(f'Checking for duplicate entries. Dropped {num_entries - len(df)} entries.')
    num_entries = len(df)
    if num_entries == 0:
        print('No entries remaining')
        return pl.DataFrame()
    print(f'Number of entries remaining: {num_entries}')
    df = df.filter(pl.col("pChEMBL_Value").is_not_null())

    return df