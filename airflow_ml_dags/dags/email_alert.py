from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.email import send_email
from datetime import timedelta
import os


def notify_email(contextDict, **kwargs):
    title = "Airflow alert: file does not exist.".format(**contextDict)
    body = """
    Hi Everyone, <br>
    <br>
    There's been an error in the data_exist job. No data available. <br>
    <br>Please, fix it.    
    """.format(**contextDict)
    send_email('muzaffar.soliyev97@gmail.com', title, body)


default_args = {
    'owner': 'airflow',
    'description': 'email_alert',
    'start_date': days_ago(1),
}


def _data_exist():
    assert True == os.path.exists("/opt/airflow/data/raw/{{ ds }}/data.csv")


with DAG(
        "email_alert_dag",
        default_args=default_args,
        description="This DAG sends email alert",
        schedule_interval=timedelta(days=1),
) as dag:
    data_exist = PythonOperator(task_id='email_alert_task',
                             python_callable=_data_exist,
                             on_failure_callback=notify_email,
                             dag=dag)

    data_exist
