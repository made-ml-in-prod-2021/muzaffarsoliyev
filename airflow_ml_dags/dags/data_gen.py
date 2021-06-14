import pathlib
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import random


def gen_random_in_range(range_tuple: tuple):
    if (isinstance(range_tuple[0], float)):
        return round(random.uniform(range_tuple[0], range_tuple[1]), 1)
    return random.randint(range_tuple[0], range_tuple[1])


def gen_synthetic_data(num_rows: int = 100) -> pd.DataFrame:
    """
    Method that creates fake data like heart-disease-uci dataset (unstratified)
    Returns fake dataset in pandas.DataFrame
    """
    ranges = {"age": (29, 77), "sex": (0, 1), "cp": (0, 3), "trestbps": (94, 200),
              "chol": (126, 564), "fbs": (0, 1), "restecg": (0, 2), "thalach": (71, 202),
              "exang": (0, 1), "oldpeak": (0.0, 6.2), "slope": (0, 2), "ca": (0, 4),
              "thal": (0, 3), "target": (0, 1)}
    column_names = [key for key, value in ranges.items()]

    df = pd.DataFrame(columns=column_names)
    for i in range(num_rows):
        row = []
        for key, value in ranges.items():
            random_value = gen_random_in_range(value)
            row.append(random_value)
        df = df.append(pd.Series(row, index=column_names), ignore_index=True)
    df = df.astype({"age": int, "sex": int, "cp": int, "trestbps": int,
                    "chol": int, "fbs": int, "restecg": int, "thalach": int,
                    "exang": int, "oldpeak": float, "slope": int, "ca": int,
                    "thal": int, "target": int})
    return df


def _gen_data():
    df = gen_synthetic_data(300)
    print(df)
    data_df = df.copy()
    data_df.drop("target", axis=1, inplace=True)
    target_df = pd.DataFrame(df["target"], columns=['target'])

    pathlib.Path("/opt/airflow/data/raw/{{ ds }}").mkdir(parents=True, exist_ok=True)
    data_df.to_csv("/opt/airflow/data/raw/{{ ds }}/data.csv", index=False)
    target_df.to_csv("/opt/airflow/data/raw/{{ ds }}/target.csv", index=False)


with DAG(
        dag_id="data_gen",
        description="This DAG generated synthetic data",
        start_date=days_ago(14),
        schedule_interval=None,  # timedelta(days=1),
) as dag:
    get_pictures = PythonOperator(
        task_id="data_gen",
        python_callable=_gen_data,
        dag=dag,
    )

    get_pictures
