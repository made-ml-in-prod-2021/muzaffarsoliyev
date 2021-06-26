import pathlib
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle
from airflow.models import Variable
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
from airflow.utils.dates import days_ago


def _preprocess_data():
    data_df = pd.read_csv("/opt/airflow/data/raw/{{ ds }}/data.csv")
    target_df = pd.read_csv("/opt/airflow/data/raw/{{ ds }}/target.csv")

    print(f"data before transform: {data_df}")
    data_df.drop(columns=["fbs"], inplace=True)
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
    transformer.fit_transform(data_df)

    data_df["target"] = target_df
    print(f"data after transform: {data_df}")

    pathlib.Path("/opt/airflow/data/processed/{{ ds }}").mkdir(parents=True, exist_ok=True)

    processed_path = "/opt/airflow/data/processed/{{ ds }}/data.csv"
    print(f"saving processed data to {processed_path}")
    data_df.to_csv(processed_path, index=False)


def _predict_model():
    data = pd.read_csv("/opt/airflow/data/processed/{{ ds }}/data.csv")
    target = data["target"]
    data.drop(columns=["target"], inplace=True)
    dag_config = Variable.get("variables_config", deserialize_json=True)
    model_path = dag_config["model_path"]
    model = pickle.load(open(model_path, "rb"))
    predicts = model.predict(data)

    pathlib.Path("/opt/airflow/data/predictions/{{ ds }}").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predicts, columns=['predictions']).to_csv("/opt/airflow/data/predictions/{{ ds }}/predictions.csv", index=None, mode='w')


with DAG(
        dag_id="predict",
        description="This DAG predicts on synthetic data using Airflow variables (model_path)",
        start_date=days_ago(0),
        schedule_interval=timedelta(days=1),
) as dag:
    preprocess_data = PythonOperator(
        task_id="data_preprocessing",
        python_callable=_preprocess_data,
        dag=dag,
    )
    predict_model = PythonOperator(
        task_id="predict_model",
        python_callable=_predict_model,
        dag=dag
    )
    preprocess_data >> predict_model
