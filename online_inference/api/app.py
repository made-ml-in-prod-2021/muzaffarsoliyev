import os
import logging
from typing import List, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conlist
from src.enities.train_pipeline_params import (
    read_training_pipeline_params,
)
from src.predict import predict_online_inference

app = FastAPI()
logger = logging.getLogger(__name__)

BAD_REQIEST_ERR_MSG = "Incorrect column ordering or incorrect data shape"
PAYLOAD_TOO_LARGE_MSG = "Your request is too large"
TOO_MUCH_DATA_CONSTRAINT = 20

model: bool = False

# set root dir as current directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("..")


# override default exception code (now 400, not 422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": BAD_REQIEST_ERR_MSG}
    )


@app.get("/")
def main():
    return "Welcome to Heart Disease UCI predictor!"


@app.get("/healz")
def health() -> bool:
    return not (model is False)


@app.on_event("startup")
def load_model_and_transformer():
    global model
    model_path = os.getenv("PATH_TO_MODEL", "models/model.pkl")
    transformer_path = os.getenv("PATH_TO_TRANSFORMER", "models/transformer.pkl")
    if model_path is None or transformer_path is None:
        err = f"PATH_TO_MODEL {model_path} or PATH_TO_TRANSFORMER {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    model = True


class PredictResponse(BaseModel):
    predictions: List[int]


class HeartDiseaseUCIModel(BaseModel):
    data: List[conlist(Union[float, int, None], min_items=13, max_items=13)]
    features: List[str]


def make_predict(
        data: List[List], features: List[str]) -> PredictResponse:
    data = pd.DataFrame(data, columns=features)
    print(os.getcwd())
    data = data.astype({"age": int, "sex": int, "cp": int, "trestbps": int,
                        "chol": int, "fbs": int, "restecg": int, "thalach": int,
                        "exang": int, "oldpeak": float, "slope": int, "ca": int,
                        "thal": int})
    params = read_training_pipeline_params("configs/train_config.yaml")
    predicts = predict_online_inference(params, data)

    return PredictResponse(predictions=predicts.tolist())


def bad_request(data: List[List], features: List[str]) -> bool:
    model_columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca", "thal"]
    columns_num = len(model_columns)
    if model_columns != features:
        return True
    for data_row in data:
        if len(data_row) != columns_num:
            return True

    return False


def too_large_request(data: List[List]) -> bool:
    return len(data) > TOO_MUCH_DATA_CONSTRAINT


@app.get("/predict/", response_model=PredictResponse)
def predict(request: HeartDiseaseUCIModel):
    if too_large_request(request.data):
        raise HTTPException(status_code=413, detail=PAYLOAD_TOO_LARGE_MSG)
    if bad_request(request.data, request.features):
        raise HTTPException(status_code=400, detail=BAD_REQIEST_ERR_MSG)
    return make_predict(request.data, request.features)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 80))
