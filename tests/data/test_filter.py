import polars as pl

from novami.data.filter import filter_outliers


def test_filter_outliers_keeps_expected_rows_and_columns():
    df = pl.DataFrame({"SMILES": ["CCO", "c1ccccc1"]})
    out = filter_outliers(df, kwargs={"MolWt": (0, 50)})
    assert out.columns == ["SMILES"]
    assert out["SMILES"].to_list() == ["CCO"]
