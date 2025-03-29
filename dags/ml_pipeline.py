from airflow import DAG
from airflow import task
from datetime import datetime
from airflow.operators.python import PythonOperator
from src.components.data_ingestion import DataIngestion


# Define the DAG
with DAG(
    dag_id = 'Whole_ml_pipeline',
    start_date = datetime(2025,1,1),
    schedule_interval= '@once',
    catchup = False,
) as dag:
    
    data_ingestion_instance = DataIngestion()
    
    data_ingest = PythonOperator(
        task_id='data_ingestion',
        python_callable=DataIngestion.initiate_data_ingestion,
        dag=dag,
    )


    
    

