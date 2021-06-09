from typing import List

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import process_categorical_features, process_numerical_features, extract_target, drop_columns
from src.enities.feature_params import FeatureParams

@pytest.fixture()
def categorical_feature() -> str:
    return "categorical_feature"


@pytest.fixture()
def categorical_values() -> List[str]:
    return ["cat", "dog", "cow"]


@pytest.fixture()
def categorical_values_with_nan(categorical_values: List[str]) -> List[str]:
    return categorical_values + [np.nan]


@pytest.fixture
def fake_categorical_data(
    categorical_feature: str, categorical_values_with_nan: List[str]
) -> pd.DataFrame:
    return pd.DataFrame({categorical_feature: categorical_values_with_nan})


def test_process_categorical_features(
    fake_categorical_data: pd.DataFrame,
    categorical_feature: str,
    categorical_values: List[str],
):
    print(fake_categorical_data)
    transformed: pd.DataFrame = process_categorical_features(fake_categorical_data)
    assert transformed.shape[1] == 3
    assert transformed.sum().sum() == 4


@pytest.fixture()
def numerical_feature() -> str:
    return "numerical_feature"


@pytest.fixture()
def numerical_values() -> List[float]:
    return [1, 2, 3]


@pytest.fixture()
def numerical_values_with_nan(numerical_values: List[float]) -> List[float]:
    return numerical_values + [np.nan]


@pytest.fixture
def fake_numerical_data(
    numerical_feature: str, numerical_values_with_nan: List[float]
) -> pd.DataFrame:
    return pd.DataFrame({numerical_feature: numerical_values_with_nan}, index=range(0, len(numerical_values_with_nan)))


def test_process_numerical_features(
    fake_numerical_data: pd.DataFrame,
):
    transformed: pd.DataFrame = process_numerical_features(fake_numerical_data)
    assert transformed.shape == fake_numerical_data.shape # shape is immutable
    assert transformed.sum().sum() == 0 # StandardScaler


@pytest.fixture()
def target_feature() -> str:
    return "target"


@pytest.fixture()
def target_values() -> List[int]:
    return [0, 1, 1]


@pytest.fixture
def fake_target_data(
    target_feature: str, target_values: List[float]
) -> pd.DataFrame:
    df = pd.DataFrame(target_values)
    df.columns = [target_feature]
    return df


@pytest.fixture
def feature_params(target_feature) -> FeatureParams:
    return FeatureParams(use_log_trick=False, target_col=target_feature, categorical_features=[], numerical_features=[], features_to_drop=[target_feature])


def test_extract_target(
    fake_target_data: pd.DataFrame,
    target_values: List[int],
    target_feature: str,
    feature_params: FeatureParams
):
    target: pd.DataFrame = extract_target(fake_target_data, feature_params)
    comparison = (target.to_numpy() == np.array(target_values))
    assert comparison.all()


def test_drop_column(
    fake_target_data: pd.DataFrame,
    target_values: List[int],
    target_feature: str,
    feature_params: FeatureParams
):
    dropped: pd.DataFrame = drop_columns(fake_target_data, feature_params)
    assert dropped.empty



