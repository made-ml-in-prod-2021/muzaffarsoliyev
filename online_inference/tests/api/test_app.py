from api.app import (
    TOO_MUCH_DATA_CONSTRAINT,
    PAYLOAD_TOO_LARGE_MSG,
    BAD_REQIEST_ERR_MSG
)
from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def get_err_message(response):
    return response.json()["detail"]


def test_predict_ok():
    response = client.get("/predict/", json={
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "data": [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    })
    assert response.status_code == 200
    response_json = response.json()
    assert "predictions" in response_json
    assert len(response_json["predictions"]) == 1


def test_predict_wrong_columns():
    response = client.get("/predict/", json={
        "features": ["something", "wrong"],
        "data": [[0, 0]]
    })
    assert response.status_code == 400
    assert get_err_message(response) == BAD_REQIEST_ERR_MSG


def test_predict_wrong_data():
    response = client.get("/predict/", json={
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "data": [[1]]
    })
    assert response.status_code == 400
    assert get_err_message(response) == BAD_REQIEST_ERR_MSG


def test_predict_too_large_payload():
    too_much_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1] for i in range(TOO_MUCH_DATA_CONSTRAINT + 1)]
    response = client.get("/predict/", json={
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "data": too_much_data
    })
    assert response.status_code == 413  # too much data
    assert get_err_message(response) == PAYLOAD_TOO_LARGE_MSG
