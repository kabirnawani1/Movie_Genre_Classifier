from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from google.cloud import storage

def handle_response(response):
    print("Response status code:", response.status_code)
    print("Response body:", response.text)
    return response.text

def WriteToGcs(data):
    # data = ti.xcom_pull(task_ids=['get_http_data_from_movies_api'])
    bucket_name = 'us-west1-msds-dds-project-53a89e6e-bucket'
    destination_blob_name = 'data/new_releases/new_movie.json'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(json.dumps(data))

    print(
        f"{destination_blob_name} with contents uploaded to {bucket_name}."
    )

dag = DAG(
    "project_msds_api_call", #dag id
    default_args={"retries": 2,
                  "retry_delay": timedelta(minutes=1),
                  },
    description="API call for MSDS Distributed systems project",
    start_date=datetime(2024, 1, 1), #start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    # end_date = "",
    schedule='@once', 
    catchup=False,
    tags=["project-msds-697"]
)

trigger_api = TriggerDagRunOperator(
        task_id='trigger_dag_API',
        trigger_dag_id='project_msds_crawler_call',  # Name of the first DAG
        dag=dag,
        execution_date="{{ execution_date }}"
    )
trigger_model = TriggerDagRunOperator(
    task_id='trigger_dag_MODEL',
    trigger_dag_id='model_dag_final',  # Name of the first DAG
    dag=dag,
    execution_date="{{ execution_date }}"
    )

read_file_task = BashOperator(
    task_id='read_files',
    bash_command='gsutil cat gs://us-west1-msds-dds-project-53a89e6e-bucket/data/new_releases/new_movie.txt',
    dag=dag)

def final_print(**kwargs):
    print("HELLO WORLD, THE PIPELINE IS OVER")

def initial_print(**kwargs):
    print("HELLO WORLD, THE PIPELINE IS ABOUT TO START")

initial_check = PythonOperator(task_id="first_thing", python_callable=initial_print)

final_check = PythonOperator(task_id="last_thing", python_callable=final_print)

http_operator = SimpleHttpOperator(
    task_id="get_http_data_from_movies_api",
    http_conn_id="movies_api",
    method="GET",
    headers={"Content-Type": "application/json"},
    endpoint="/?apikey=a1fbb45d&t={{ti.xcom_pull(task_ids='read_files')}}&plot=full",
    response_filter=handle_response,  # Pass your custom function to handle the response
    dag=dag
)
write_data_to_gcs = PythonOperator(
    task_id = 'write_data_to_gcs',
    python_callable = WriteToGcs,
    op_args=[http_operator.output],
    dag=dag
)
# Task dependency set
initial_check >> trigger_api >> read_file_task >> http_operator >> write_data_to_gcs >> trigger_model >> final_check 
