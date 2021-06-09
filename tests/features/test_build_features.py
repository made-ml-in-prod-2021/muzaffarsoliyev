from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.data.make_dataset import read_data
from src.enities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer

from ..data import test_make_dataset, data, DEFAULT_GENERATED_DATA_NAME, DEFAULT_DATA_SIZE, COLUMN_COUNT
from .. import gen_synthetic_data


@pytest.fixture()
def categorical_features() -> str:
    return ["sex", "cp", "restecg", "exang",  "slope", "ca", "thal"]


@pytest.fixture()
def features_to_drop() -> str:
    return ["fbs"]


@pytest.fixture()
def numerical_features() -> str:
    return ["age", "trestbps", "chol", "thalach","oldpeak"]


@pytest.fixture()
def target_col() -> str:
    return "target"


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
        use_log_trick=True,
    )
    return params


def test_make_features(
    feature_params: FeatureParams, data: pd.DataFrame,
):
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in feature_params.features_to_drop)


def test_extract_features(feature_params: FeatureParams, data: pd.DataFrame):
    target = extract_target(data, feature_params)
    assert_allclose(
        np.log(data[feature_params.target_col].to_numpy()), target.to_numpy()
    )

