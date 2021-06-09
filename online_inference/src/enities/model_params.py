from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class LogisticRegressionParams:
    C: float = field(default=1.0)
    solver: str = field(default="liblinear")
    max_iter: int = field(default=128)


@dataclass
class DecisionTreeClassifierParams:
    criterion: str = field(default="entropy")
    max_depth: int = field(default=128)


LogisticRegressionParamsSchema = class_schema(LogisticRegressionParams)
DecisionTreeClassifierParamsSchema = class_schema(DecisionTreeClassifierParams)


def read_model_logistic_regression_params(path: str) -> LogisticRegressionParams:
    with open(f'{path}/../model_configs/logistic_regression.yaml', "r") as input_stream:
        schema = LogisticRegressionParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


def read_model_decision_tree_classifier_params(path: str) -> DecisionTreeClassifierParams:
    with open(f'{path}/../model_configs/decision_tree_classifier.yaml', "r") as input_stream:
        schema = DecisionTreeClassifierParamsSchema()
        return schema.load(yaml.safe_load(input_stream))