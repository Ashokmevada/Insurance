from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'Ashok Mevada',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='insurance_pipeline_dvc_dag',
    default_args=default_args,
    description='Run Insurance ML pipeline using DVC every 5 minutes',
    schedule_interval='*/5 * * * *',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['dvc', 'insurance', 'ml']
) as dag:

    dvc_pull = BashOperator(
        task_id='dvc_pull',
        bash_command='cd /usr/local/airflow/ && dvc pull'
    )

    data_ingestion = BashOperator(
        task_id='data_ingestion',
        bash_command='cd /usr/local/airflow/ && dvc repro data_ingestion'
    )

    data_validation = BashOperator(
        task_id='data_validation',
        bash_command='cd /usr/local/airflow/ && dvc repro data_validation'
    )

    data_transformation = BashOperator(
        task_id='data_transformation',
        bash_command='cd /usr/local/airflow/ && dvc repro data_transformation'
    )

    model_training = BashOperator(
        task_id='model_training',
        bash_command='cd /usr/local/airflow/ && dvc repro model_training'
    )

    # DAG execution order
    dvc_pull >> data_ingestion >> data_validation >> data_transformation >> model_training
