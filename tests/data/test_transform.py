import numpy as np
import polars as pl
import pytest

from novami.data.transform import DataTransformer, get_transformer_params


def test_data_transformer_validate_helpers():
    x = DataTransformer.validate_features(np.array([1.0, 2.0]))
    y = DataTransformer.validate_targets(np.array([[1.0], [2.0]]))
    assert x.shape == (1, 2)
    assert y.shape == (2,)

    with pytest.raises(TypeError):
        DataTransformer.validate_features([1.0, 2.0])


def test_data_transformer_fit_transform_pipeline():
    x = np.array(
        [
            [1.0, 0.0, 2.0],
            [2.0, 0.0, np.nan],
            [3.0, 0.0, 4.0],
            [4.0, 0.0, 5.0],
        ]
    )
    transformer = DataTransformer(use_masks=True, use_corr=False, use_imputer=True, use_selector=True, use_scaler=True)
    out = transformer.fit_transform(x)
    assert out.ndim == 2
    assert out.shape[0] == x.shape[0]
    assert np.isfinite(out).all()


def test_data_transformer_transform_df_and_fit_transform_df():
    df = pl.DataFrame({"X": [np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([3.0, 4.0])]})
    transformer = DataTransformer(use_masks=True, use_corr=False, use_imputer=True, use_selector=False, use_scaler=False)
    fit_df = transformer.fit_transform_df(df, features_col="X")
    out_df = transformer.transform_df(df, features_col="X")
    assert fit_df.height == df.height
    assert out_df.height == df.height
    assert np.asarray(fit_df["X"][0]).size == 2


def test_get_transformer_params():
    params = get_transformer_params("ECFP")
    assert params["use_masks"] is True
    with pytest.raises(KeyError):
        get_transformer_params("UNKNOWN_FEATURE")
