import os
import pickle
from typing import List, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression

from src.data.make_dataset import read_data
from src.enities import TrainingParams
from src.enities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer
from src.models.model_fit_predict import train_model, serialize_model

from ..features import categorical_features, features_to_drop, numerical_features, target_col, feature_params
from ..data import data

@pytest.fixture
def features_and_target(
    data: pd.DataFrame, categorical_features: List[str], numerical_features: List[str], feature_params: FeatureParams
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_params.use_log_trick=False
    target = extract_target(data, feature_params)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    return features, target


@pytest.fixture
def train_params() -> TrainingParams:
    params = TrainingParams(
        model_type="LogisticRegression",
        random_state=123
    )
    return params


@pytest.fixture
def model() -> LogisticRegression:
    return LogisticRegression()


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series], model: LogisticRegression):
    features, target = features_and_target
    model = train_model(features, target, model)
    assert isinstance(model, LogisticRegression)
    assert model.predict(features).shape[0] == target.shape[0]





def test_serialize_model(tmpdir: LocalPath, model: LogisticRegression):
    expected_output = tmpdir.join("model.pkl")
    model = LogisticRegression()
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, LogisticRegression)