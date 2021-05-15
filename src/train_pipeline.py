import json
import logging
import logging.config
import yaml
import sys

import click
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.data import read_data, split_train_val_data
from src.enities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from src.enities.model_params import read_model_logistic_regression_params, read_model_decision_tree_classifier_params
from src.features import make_features
from src.features.build_features import extract_target, build_transformer, drop_columns
from src.models import train_model, serialize_model, predict_model, evaluate_model, SklearnClassifierModel

NOT_ENOUGH_DATA_THRESHOLD = 50

APPLICATION_NAME = "ml_project"
APPLICATION_NAME_WARNING = "ml_project_warning"
DEFAULT_LOGGING_CONFIG_PATH = "configs/logging_config.yml"
logger = logging.getLogger(APPLICATION_NAME)
warning_logger = logging.getLogger(APPLICATION_NAME_WARNING)


def train_pipeline(training_pipeline_params: TrainingPipelineParams, model: SklearnClassifierModel):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    data = drop_columns(data, training_pipeline_params.feature_params)
    logger.info(f"data.shape after dropping some columns is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    if train_df.shape[0] < NOT_ENOUGH_DATA_THRESHOLD:
        msg = "No enough data to build good model"
        logger.warning(msg)
        warning_logger.warning(msg)

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, model
    )

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(
        model,
        val_features,
        training_pipeline_params.feature_params.use_log_trick,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
        use_log_trick=training_pipeline_params.feature_params.use_log_trick,
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    model_type = params.train_params.model_type

    logger.info(f"model is {model_type}")

    if model_type == "LogisticRegression":
        model_params = read_model_logistic_regression_params(config_path)
        model = LogisticRegression(
            C=model_params.C, solver=model_params.solver, max_iter=model_params.max_iter, random_state=params.train_params.random_state
        )
    elif model_type == "DecisionTreeClassifier":
        model_params = read_model_decision_tree_classifier_params(config_path)
        model = DecisionTreeClassifier(
            criterion=model_params.criterion, max_depth=model_params.max_depth, random_state=params.train_params.random_state)
    else:
        raise NotImplementedError()

    train_pipeline(params, model)


def setup_logging():
    with open(DEFAULT_LOGGING_CONFIG_PATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


if __name__ == "__main__":
    setup_logging()
    train_pipeline_command()
