import numpy as np
import polars as pl
import pytest

from novami.data.descriptors import (
    dataframe_2_atompair,
    dataframe_2_daylight,
    dataframe_2_ecfp,
    dataframe_2_maccs,
    dataframe_2_rdkit,
    smiles_2_atompair,
    smiles_2_daylight,
    smiles_2_ecfp,
    smiles_2_maccs,
    smiles_2_rdkit,
)


VALID_SMILES = "CCO"
OTHER_VALID_SMILES = "c1ccccc1"
INVALID_SMILES = "not-a-smiles"


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (smiles_2_ecfp, {"nbits": 128}),
        (smiles_2_daylight, {"nbits": 128}),
        (smiles_2_atompair, {"nbits": 128}),
        (smiles_2_maccs, {}),
        (smiles_2_rdkit, {"decimals": 5}),
    ],
)
def test_smiles_functions_single_valid_returns_array(func, kwargs):
    result = func(VALID_SMILES, **kwargs)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.size > 0


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (smiles_2_ecfp, {"nbits": 128}),
        (smiles_2_daylight, {"nbits": 128}),
        (smiles_2_atompair, {"nbits": 128}),
        (smiles_2_maccs, {}),
        (smiles_2_rdkit, {"decimals": 5}),
    ],
)
def test_smiles_functions_list_valid_returns_list_of_arrays(func, kwargs):
    result = func([VALID_SMILES, OTHER_VALID_SMILES], **kwargs)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(entry, np.ndarray) for entry in result)
    assert all(entry.ndim == 1 for entry in result)
    assert all(entry.size > 0 for entry in result)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (smiles_2_ecfp, {"nbits": 128}),
        (smiles_2_daylight, {"nbits": 128}),
        (smiles_2_atompair, {"nbits": 128}),
        (smiles_2_maccs, {}),
        (smiles_2_rdkit, {"decimals": 5}),
    ],
)
def test_smiles_functions_invalid_string_returns_nan(func, kwargs):
    result = func(INVALID_SMILES, **kwargs)
    assert np.isnan(result)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (smiles_2_ecfp, {"nbits": 128}),
        (smiles_2_daylight, {"nbits": 128}),
        (smiles_2_atompair, {"nbits": 128}),
        (smiles_2_maccs, {}),
        (smiles_2_rdkit, {"decimals": 5}),
    ],
)
def test_smiles_functions_mixed_list_returns_all_nan(func, kwargs):
    result = func([VALID_SMILES, INVALID_SMILES], **kwargs)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(np.isnan(entry) for entry in result)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (smiles_2_ecfp, {"nbits": 128}),
        (smiles_2_daylight, {"nbits": 128}),
        (smiles_2_atompair, {"nbits": 128}),
        (smiles_2_maccs, {}),
        (smiles_2_rdkit, {"decimals": 5}),
    ],
)
def test_smiles_functions_unsupported_input_raises_type_error(func, kwargs):
    with pytest.raises(TypeError):
        func(12345, **kwargs)


@pytest.mark.parametrize(
    "func",
    [smiles_2_ecfp, smiles_2_daylight, smiles_2_atompair],
)
def test_count_mode_shape_and_numeric_properties(func):
    bits_result = func(VALID_SMILES, nbits=128, count=False)
    count_result = func(VALID_SMILES, nbits=128, count=True)

    assert isinstance(bits_result, np.ndarray)
    assert isinstance(count_result, np.ndarray)
    assert bits_result.shape == count_result.shape == (128,)
    assert np.all(count_result >= 0)


@pytest.mark.parametrize(
    "func",
    [smiles_2_ecfp, smiles_2_daylight, smiles_2_atompair],
)
def test_count_mode_single_vs_batch_regression(func):
    single = func(VALID_SMILES, nbits=128, count=True)
    batched = func([VALID_SMILES], nbits=128, count=True)[0]

    assert isinstance(single, np.ndarray)
    assert isinstance(batched, np.ndarray)
    assert single.shape == batched.shape
    # Keep this assertion stable across RDKit/count-vector representations.
    # Single and batched calls should at least agree on which bins are present.
    assert np.array_equal(single > 0, batched > 0)


@pytest.mark.parametrize(
    ("func", "default_col", "kwargs"),
    [
        (dataframe_2_ecfp, "ECFP", {"nbits": 128, "count": False}),
        (dataframe_2_daylight, "Daylight", {"nbits": 128, "count": False}),
        (dataframe_2_atompair, "AtomPair", {"nbits": 128, "count": False}),
        (dataframe_2_maccs, "MACCS", {}),
        (dataframe_2_rdkit, "RDKit", {"decimals": 5}),
    ],
)
def test_dataframe_wrappers_add_default_descriptor_and_preserve_alignment(func, default_col, kwargs):
    df = pl.DataFrame(
        {
            "SMILES": [VALID_SMILES, OTHER_VALID_SMILES, VALID_SMILES],
            "row_id": [0, 1, 2],
        }
    )

    out = func(df=df, n_jobs=1, batch_size=1, **kwargs)

    assert out.height == df.height
    assert out["SMILES"].to_list() == df["SMILES"].to_list()
    assert out["row_id"].to_list() == df["row_id"].to_list()
    assert default_col in out.columns
    assert out[default_col].is_null().sum() == 0


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (dataframe_2_ecfp, {"nbits": 128, "count": False}),
        (dataframe_2_daylight, {"nbits": 128, "count": False}),
        (dataframe_2_atompair, {"nbits": 128, "count": False}),
        (dataframe_2_maccs, {}),
        (dataframe_2_rdkit, {"decimals": 5}),
    ],
)
def test_dataframe_wrappers_support_custom_descriptor_col(func, kwargs):
    df = pl.DataFrame(
        {
            "SMILES": [VALID_SMILES, OTHER_VALID_SMILES, VALID_SMILES],
        }
    )

    out = func(df=df, descriptor_col="CUSTOM_DESC", n_jobs=1, batch_size=1, **kwargs)

    assert "CUSTOM_DESC" in out.columns
    assert out["CUSTOM_DESC"].is_null().sum() == 0


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (dataframe_2_ecfp, {"nbits": 128, "count": True}),
        (dataframe_2_daylight, {"nbits": 128, "count": True}),
        (dataframe_2_atompair, {"nbits": 128, "count": True}),
    ],
)
def test_dataframe_wrappers_count_mode_default_column_name(func, kwargs):
    df = pl.DataFrame(
        {
            "SMILES": [VALID_SMILES, OTHER_VALID_SMILES],
        }
    )

    out = func(df=df, n_jobs=1, batch_size=1, **kwargs)
    expected_col = (
        "ECFPCount"
        if func is dataframe_2_ecfp
        else "DaylightCount"
        if func is dataframe_2_daylight
        else "AtomPairCount"
    )
    assert expected_col in out.columns
    assert out[expected_col].is_null().sum() == 0
