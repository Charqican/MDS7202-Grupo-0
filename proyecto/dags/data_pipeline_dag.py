import os
import sys
from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# Asegura que /opt/airflow/scripts esté en PYTHONPATH
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
scripts_path = os.path.join(AIRFLOW_HOME, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Importa las funciones de transformations.py
def import_transformers():
    global extract_and_merge, create_label, prepare_model_dataset, transform_features, split_train_val_windows
    from transformations import (
        extract_and_merge,
        create_label,
        prepare_model_dataset,
        transform_features,
        split_train_val_windows
    )
import_transformers()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
}

with DAG(
    dag_id='data_pipeline',
    default_args=default_args,
    start_date=datetime(2025, 6, 19),
    schedule_interval=None,
    catchup=False,
    tags=['etl', 'preprocessing']
) as dag:

    def task_extract(**kwargs):
        merge_path = extract_and_merge()
        kwargs['ti'].xcom_push(key='merge_path', value=merge_path)

    def task_label(**kwargs):
        ti = kwargs['ti']
        merge_path = ti.xcom_pull(task_ids='extract', key='merge_path')
        label_path = create_label(merge_path)
        ti.xcom_push(key='label_path', value=label_path)

    def task_prepare(**kwargs):
        ti = kwargs['ti']
        merge_path = ti.xcom_pull(task_ids='extract', key='merge_path')
        label_path = ti.xcom_pull(task_ids='label', key='label_path')
        dataset_path = prepare_model_dataset(merge_path, label_path)
        ti.xcom_push(key='dataset_path', value=dataset_path)

    def task_split(**kwargs):
        ti = kwargs['ti']
        dataset_path = ti.xcom_pull(task_ids='prepare', key='dataset_path')
        splits = split_train_val_windows(dataset_path, time_window=4, stride=3)
        # Guarda cada split en archivos individuales
        processed_dir = os.path.join(AIRFLOW_HOME, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        split_paths = {'train': [], 'val': []}
        for idx, (train_df, val_df) in enumerate(splits):
            train_file = os.path.join(processed_dir, f'train_{idx}.parquet')
            val_file   = os.path.join(processed_dir, f'val_{idx}.parquet')
            train_df.to_parquet(train_file, index=False)
            val_df.to_parquet(val_file, index=False)
            split_paths['train'].append(train_file)
            split_paths['val'].append(val_file)
        ti.xcom_push(key='split_paths', value=split_paths)

    def task_transform(**kwargs):
        ti = kwargs['ti']
        dataset_path = ti.xcom_pull(task_ids='prepare', key='dataset_path')
        df = pd.read_parquet(dataset_path)
        df_feat = transform_features(df)
        # Guarda features
        processed_dir = os.path.join(AIRFLOW_HOME, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        out_path = os.path.join(processed_dir, 'features.parquet')
        df_feat.to_parquet(out_path, index=False)
        ti.xcom_push(key='features_path', value=out_path)

    extract = PythonOperator(
        task_id='extract',
        python_callable=task_extract
    )

    label = PythonOperator(
        task_id='label',
        python_callable=task_label
    )

    prepare = PythonOperator(
        task_id='prepare',
        python_callable=task_prepare
    )
    split = PythonOperator(
        task_id='split',
        python_callable=task_split
    )
    transform = PythonOperator(
        task_id='transform',
        python_callable=task_transform
    )

    extract >> label >> prepare >> split >> transform

data_pipeline = dag
