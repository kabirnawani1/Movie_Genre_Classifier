from __future__ import annotations
from datetime import datetime, timedelta

from lxml import etree

from airflow import DAG
from airflow.providers.google.cloud.operators.gcs import GCSDeleteObjectsOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import PythonOperator
from google.cloud import storage
import random

from urllib.parse import quote


def WriteToGcs(data):
    # data = ti.xcom_pull(task_ids=['get_http_data_from_movies_api'])
    bucket_name = 'us-west1-msds-dds-project-53a89e6e-bucket'
    destination_blob_name = 'data/new_releases/new_movie.txt'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    # data= ','.join([item for item in data])
    print(data)
    blob.upload_from_string(data)
    print(
        f"{destination_blob_name} with contents uploaded to {bucket_name}."
    )

def handle_scrapper_response(response,**kwargs):
    tree = etree.HTML(response.text)
    
    table = tree.xpath('.//div[@class="a-section imdb-scroll-table-inner"]')
    if len(table)==0:
        return False
    else:
        new_releases = tree.xpath("//tr[contains(@class, 'isNewThisWeek')]/td[3]/a/text()")
        new_releases = [(quote(item.encode('utf-8'))).replace(' ','+') for item in new_releases]
        print("Response status code:", response.status_code)
        print("Response result:", new_releases)
        return new_releases[0]
    

with DAG(
    "project_msds_crawler_call", #dag id
    default_args={'retries': 3,  # Number of retries for the task
                  'retry_delay': timedelta(seconds=5)
                  },
    description="API call for MSDS Distributed systems project",
    start_date=datetime(2024, 1, 1), #start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    # end_date = "",
    schedule_interval="@weekly",  # Setting schedule_interval to None to prevent automatic runs
    catchup=False,
    tags=["project-msds-697"]
) as dag:
    # Define the SimpleHttpOperator

    current_date = datetime.now()
    year = 2023
    week = str(random.randint(1, 52)).zfill(2) # str(current_date.isocalendar()[1]-2).zfill(2)
    
    cleanup = GCSDeleteObjectsOperator(
            task_id="cleanup",
            bucket_name='us-west1-msds-dds-project-53a89e6e-bucket',
            prefix=f"data/new_releases/",
            execution_timeout=timedelta(minutes=5),
        )
    
    http_scrapper = SimpleHttpOperator(
        task_id="scrape_new_releases",
        http_conn_id="new_releases_api",
        method="GET",
        headers={"Content-Type": "application/json"},
        endpoint=f"""/weekly/{year}W{week}/?ref_=bo_wl_nav""",
        response_filter=handle_scrapper_response,  # Pass your custom function to handle the response
        dag=dag
    )

    write_data_to_gcs = PythonOperator(
        task_id = 'write_data_to_gcs',
        python_callable = WriteToGcs,
        op_args=[http_scrapper.output],
        dag=dag
    )

cleanup >> http_scrapper >> write_data_to_gcs
