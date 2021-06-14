from datetime import timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor

from airflow.utils.dates import days_ago


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(2),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _data_ready_for_train():
    return os.path.exists("/opt/airflow/data/processed/{{ ds }}/data.csv")


def _data_ready_for_predict():
    return os.path.exists("/opt/airflow/data/raw/{{ ds }}/data.csv")


with DAG(
    "data_ready_sensor",
    default_args=default_args,
    description="This DAG checks that data is ready",
    schedule_interval=timedelta(days=1),
) as dag:
    wait_data_ready_for_train = PythonSensor(
        task_id="data_ready_for_train",
        python_callable=_data_ready_for_train,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    wait_data_ready_for_predict = PythonSensor(
        task_id="data_ready_for_predict",
        python_callable=_data_ready_for_predict,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    t = BashOperator(
        task_id="touch_file",
        bash_command="touch /opt/airflow/data/ready.txt",
    )

    wait_data_ready_for_train >> wait_data_ready_for_predict >> t