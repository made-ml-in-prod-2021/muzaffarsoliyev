from src.data.make_dataset import read_data, split_train_val_data
from src.enities import SplittingParams
from .. import gen_synthetic_data
import os
import pytest

TARGET_COL_NAME = "target"
DEFAULT_GENERATED_DATA_NAME = "synthetic_heart.csv"
DEFAULT_DATA_SIZE = 200
COLUMN_COUNT = 14


@pytest.fixture
def data():
    generated_data = gen_synthetic_data(200)
    generated_data.to_csv(DEFAULT_GENERATED_DATA_NAME, index=False)
    data = read_data(DEFAULT_GENERATED_DATA_NAME)
    os.remove(DEFAULT_GENERATED_DATA_NAME)
    return data


def test_read_data_csv():
    generated_data = gen_synthetic_data(DEFAULT_DATA_SIZE)
    generated_data.to_csv(DEFAULT_GENERATED_DATA_NAME, index=False)
    data = read_data(DEFAULT_GENERATED_DATA_NAME)
    assert len(data) == DEFAULT_DATA_SIZE
    assert data.shape[1] == COLUMN_COUNT
    assert TARGET_COL_NAME in data.keys()
    os.remove(DEFAULT_GENERATED_DATA_NAME)


def test_split_train_val_data(data):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=1234, val_size=val_size,)
    train, val = split_train_val_data(data, splitting_params)
    assert val_size * train.shape[0] - val.shape[0] < 2
    assert train.shape[0] < DEFAULT_DATA_SIZE
    assert train.shape[0] > 5
