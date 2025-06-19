import os
from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
RAW_DIR       = os.path.join(AIRFLOW_HOME, 'data', 'raw')
PROCESSED_DIR = os.path.join(AIRFLOW_HOME, 'data', 'processed')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
}

with DAG(
    dag_id='extract_and_merge',
    default_args=default_args,
    start_date=datetime(2025, 6, 18),
    schedule_interval=None,
    catchup=False,
    tags=['ingest']
) as dag:

    def ingest_and_merge(**ctx):
        df_trans = pd.read_parquet(os.path.join(RAW_DIR, 'transacciones.parquet'))
        df_cli   = pd.read_parquet(os.path.join(RAW_DIR, 'clientes.parquet'))
        df_prod  = pd.read_parquet(os.path.join(RAW_DIR, 'productos.parquet'))

        df_merged = (
            df_trans
            .merge(df_cli,  on='customer_id', how='left')
            .merge(df_prod, on='product_id',  how='left')
        )

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, 'merged.parquet')
        df_merged.to_parquet(out_path, index=False)
        ctx['ti'].xcom_push('merged_path', out_path)

    merge_task = PythonOperator(
        task_id='ingest_and_merge',
        python_callable=ingest_and_merge,
    )

    merge_task
