import pathlib
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
from airflow.utils.dates import days_ago


def _preprocess_data():
    data_df = pd.read_csv("/opt/airflow/data/raw/{{ ds }}/data.csv")
    target_df = pd.read_csv("/opt/airflow/data/raw/{{ ds }}/target.csv")

    print(f"data before transform: {data_df}")
    data_df.drop(columns=["fbs"], inplace=True)
    data_df["target"] = target_df
    print(f"data after transform: {data_df}")

    pathlib.Path("/opt/airflow/data/processed/{{ ds }}").mkdir(parents=True, exist_ok=True)

    processed_path = "/opt/airflow/data/processed/{{ ds }}/data.csv"
    print(f"saving processed data to {processed_path}")
    data_df.to_csv(processed_path, index=False)


def _train_val_split():
    data = pd.read_csv("/opt/airflow/data/processed/{{ ds }}/data.csv")
    train_data, test_data = train_test_split(data, train_size=0.8)
    train_data.to_csv("/opt/airflow/data/processed/{{ ds }}/train.csv", index=False)
    test_data.to_csv("/opt/airflow/data/processed/{{ ds }}/test.csv", index=False)


def _train_model():
    train_data = pd.read_csv("/opt/airflow/data/processed/{{ ds }}/train.csv")
    target = train_data["target"]
    train_data.drop(columns=["target"], inplace=True)
    transformer = ColumnTransformer(
        [
            (
                'num',
                Pipeline([('scaler', StandardScaler())]),
                ["age", "trestbps", "chol", "thalach", "oldpeak"],
            ),
            (
                'cat',
                Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]),
                ["sex", "cp", "restecg", "exang", "slope", "ca", "thal"],
            ),
        ]
    )
    transformer.fit_transform(train_data)

    model = LogisticRegression()
    model.fit(train_data, target)

    pathlib.Path("/opt/airflow/data/models/{{ ds }}").mkdir(parents=True, exist_ok=True)
    with open("/opt/airflow/data/models/{{ ds }}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("/opt/airflow/data/models/{{ ds }}/transformer.pkl", "wb") as f:
        pickle.dump(transformer, f)


def _test_model():
    test_data = pd.read_csv("/opt/airflow/data/processed/{{ ds }}/test.csv")

    target = test_data["target"]
    test_data.drop(columns=["target"], inplace=True)
    model = pickle.load(open("/opt/airflow/data/models/{{ ds }}/model.pkl", "rb"))
    transformer = pickle.load(open("/opt/airflow/data/models/{{ ds }}/transformer.pkl", "rb"))
    transformer.transform(test_data)
    predicts = model.predict(test_data)

    metrics = {
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }

    pathlib.Path("/opt/airflow/data/metrics/{{ ds }}").mkdir(parents=True, exist_ok=True)
    with open("/opt/airflow/data/metrics/{{ ds }}/metrics.json", "w") as metric_file:
        json.dump(metrics, metric_file)


with DAG(
        dag_id="model_train",
        description="This DAG trains model on synthetic data",
        start_date=days_ago(0),
        schedule_interval=timedelta(days=1),
) as dag:
    preprocess_data = PythonOperator(
        task_id="data_preprocessing",
        python_callable=_preprocess_data,
        dag=dag,
    )
    train_val_split = PythonOperator(
        task_id="split_data",
        python_callable=_train_val_split,
        dag=dag
    )
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        dag=dag
    )
    test_model = PythonOperator(
        task_id="test_model",
        python_callable=_test_model,
        dag=dag
    )
    preprocess_data >> train_val_split >> train_model >> test_model
