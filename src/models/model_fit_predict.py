import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


SklearnClassifierModel = Union[LogisticRegression, DecisionTreeClassifier]


def train_model(
    features: pd.DataFrame, target: pd.Series, model: SklearnClassifierModel
) -> SklearnClassifierModel:
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnClassifierModel, features: pd.DataFrame, use_log_trick: bool = True
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }


def serialize_model(model: SklearnClassifierModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
