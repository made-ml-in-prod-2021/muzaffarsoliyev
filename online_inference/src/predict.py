import json
import os
import logging
import logging.config
import yaml
import sys
import pickle
from pathlib import Path
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data import read_data, split_train_val_data
from src.enities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from src.features import make_features
from src.features.build_features import extract_target, build_transformer, drop_columns
from src.models import predict_model


APPLICATION_NAME = "online_inference"
logger = logging.getLogger(APPLICATION_NAME)


def predict_pipeline(training_pipeline_params):
    path = os.getcwd()
    model = pickle.load(open(training_pipeline_params.pretrained_model_path, 'rb'))
    logger.info(f"pretrained model {model} extracted")

    logger.info(f"start predict pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    data = drop_columns(data, training_pipeline_params.feature_params)
    logger.info(f"data.shape after dropping some columns is {data.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(data)
    pred_features = make_features(transformer, data)

    predicts = predict_model(
        model,
        pred_features,
        training_pipeline_params.feature_params.use_log_trick,
    )
    predictions_path = training_pipeline_params.predictions_path
    pd.DataFrame(predicts, columns=['predictions']).to_csv(predictions_path, index=None, mode='w')
    logger.info(f"predictions are written to {predictions_path}")

    return predicts


@hydra.main(config_path="../configs", config_name="train_config.yaml")
def predict_pipeline_command(cfg: DictConfig):
    os.chdir('../../../')
    predict_pipeline(cfg)


if __name__ == "__main__":
    predict_pipeline_command()