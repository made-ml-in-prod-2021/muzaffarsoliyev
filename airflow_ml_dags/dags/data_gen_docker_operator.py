from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(2),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "docker_operator_data_gen",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    data_gen = DockerOperator(
        image="data-gen-docker-operator",
        command="/data/raw/{{ ds }}",
        task_id="docker-airflow-data-gen",
        do_xcom_push=False,
        volumes=["/Users/user/PycharmProjects/airflow_examples/airflow_examples/data:/data"]
    )

    data_gen
