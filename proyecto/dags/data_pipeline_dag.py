import os
import sys
from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configurar PYTHONPATH para scripts
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
scripts_path = os.path.join(AIRFLOW_HOME, 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Importar funciones de transformations.py
def import_transformers():
    global extract_and_merge, create_label, prepare_model_dataset, transform_features, split_train_val_windows, save_weekly_features_labels
    from scripts.transformations import (
        extract_and_merge,
        create_label,
        prepare_model_dataset,
        transform_features,
        split_train_val_windows,
        save_weekly_features_labels
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

    def task_transform(**kwargs):
        ti = kwargs['ti']
        dataset_path = ti.xcom_pull(task_ids='prepare', key='dataset_path')
        df = pd.read_parquet(dataset_path)
        df_feat = transform_features(df)
        processed_dir = os.path.join(AIRFLOW_HOME, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        features_path = os.path.join(processed_dir, 'features.parquet')
        df_feat.to_parquet(features_path, index=False)
        ti.xcom_push(key='features_path', value=features_path)

    def task_split(**kwargs):
        ti = kwargs['ti']
        features_path = ti.xcom_pull(task_ids='transform', key='features_path')
        week_files = split_train_val_windows(features_path)
        ti.xcom_push(key='week_files', value=week_files)


    def task_save_weekly(**kwargs):
        ti = kwargs['ti']
        dataset_path = ti.xcom_pull(task_ids='prepare', key='dataset_path')
        # Esto crea data/processed/features/ y data/processed/labels/
        weekly = save_weekly_features_labels(dataset_path)
        ti.xcom_push(key='weekly_files', value=weekly)
   

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

    transform = PythonOperator(
        task_id='transform',
        python_callable=task_transform
    )

    split = PythonOperator(
        task_id='split',
        python_callable=task_split
    )

    save_weekly = PythonOperator(
        task_id='save_weekly',
        python_callable=task_save_weekly)     

    # Definir orden: primero extract, label, prepare, luego transform, y finalmente split
    extract >> label >> prepare >> save_weekly >> transform >> split

data_pipeline = dag
