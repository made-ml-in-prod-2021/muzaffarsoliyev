import os
from typing import List
import pytest
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression
from src.train_pipeline import train_pipeline
from src.models.model_fit_predict import serialize_model

from src.predict import predict_pipeline
from src.enities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)
from ..tests.features import categorical_features, features_to_drop, numerical_features, target_col, feature_params
from ..tests.data import DEFAULT_GENERATED_DATA_NAME
from ..tests import gen_synthetic_data


@pytest.fixture
def dataset_path(tmpdir: LocalPath) -> str:
    generated_data = gen_synthetic_data(300)
    generated_data.to_csv(DEFAULT_GENERATED_DATA_NAME, index=False)
    return DEFAULT_GENERATED_DATA_NAME



def test_predict(
    tmpdir: LocalPath,
    dataset_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    expected_pretrained_model_path = expected_output_model_path
    expected_predictions_path = tmpdir.join("predictions.csv")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        pretrained_model_path=expected_pretrained_model_path,
        predictions_path=expected_predictions_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=1234),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
            use_log_trick=False,
        ),
        train_params=TrainingParams(model_type="LogisticRegression"),
    )
    model = LogisticRegression()
    real_model_path, metrics = train_pipeline(params, LogisticRegression())
    predicts = predict_pipeline(params)
    assert predicts[0] == 0 or predicts[0] == 1
    assert os.path.exists(real_model_path)
    assert os.path.exists(expected_predictions_path)
