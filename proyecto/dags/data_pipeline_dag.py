from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import logging
import sys

# Configurar logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------- IMPORTACIONES --------
def get_scripts_base_path():
    airflow_home = os.environ.get('AIRFLOW_HOME')
    if airflow_home:
        return os.path.join(airflow_home, 'scripts')
    return os.path.join(os.getcwd(), 'scripts')

def dynamic_import_from_path(module_name: str, base_path: str):
    import importlib.util
    file_path = os.path.join(base_path, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Cargar funciones del módulo transformations.py
scripts_path = get_scripts_base_path()
transformations = dynamic_import_from_path('transformations', scripts_path)

extract_and_merge = transformations.extract_and_merge
create_label = transformations.create_label
prepare_model_dataset = transformations.prepare_model_dataset
transform_features = transformations.transform_features
split_train_val_windows = transformations.split_train_val_windows
save_weekly_features_labels = transformations.save_weekly_features_labels

# -------- DAG DEFINITION --------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
}

with DAG(
    dag_id='etl_pipeline_features_labels',
    default_args=default_args,
    start_date=datetime(2025, 6, 29),
    schedule_interval=None,
    catchup=False,
    tags=['etl', 'features_labels']
) as dag:

    def task_extract(**kwargs):
        path = extract_and_merge()
        kwargs['ti'].xcom_push(key='merge_path', value=path)

    def task_label(**kwargs):
        ti = kwargs['ti']
        merge_path = ti.xcom_pull(task_ids='extract', key='merge_path')
        label_path = create_label(merge_path)
        ti.xcom_push(key='label_path', value=label_path)

    def task_prepare(**kwargs):
        ti = kwargs['ti']
        merge_path = ti.xcom_pull(task_ids='extract', key='merge_path')
        label_path = ti.xcom_pull(task_ids='label', key='label_path')
        prepared_path = prepare_model_dataset(merge_path, label_path)
        ti.xcom_push(key='prepared_path', value=prepared_path)

    def task_transform(**kwargs):
        ti = kwargs['ti']
        prepared_path = ti.xcom_pull(task_ids='prepare', key='prepared_path')
        transformed_path = transform_features(pd.read_parquet(prepared_path))
        ti.xcom_push(key='transformed_path', value=transformed_path)

    def task_split(**kwargs):
        ti = kwargs['ti']
        transformed_path = ti.xcom_pull(task_ids='transform', key='transformed_path')  # mismo input que transform_features
        weekly_paths = split_train_val_windows(transformed_path)
        ti.xcom_push(key='weekly_files', value=weekly_paths)

    def task_separate(**kwargs):
        ti = kwargs['ti']
        weekly_paths = ti.xcom_pull(task_ids='split', key='weekly_files')
        result = save_weekly_features_labels(weekly_paths)
        ti.xcom_push(key='separated', value=result)

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

    separate = PythonOperator(
        task_id='separate',
        python_callable=task_separate
    )

    extract >> label >> prepare >> transform >> split >> separate
