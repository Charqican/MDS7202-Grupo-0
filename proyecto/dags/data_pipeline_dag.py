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

# Funciones importadas
get_unique_customer_product_pairs = transformations.get_unique_customer_product_pairs
merge_with_features = transformations.merge_with_features
create_labels_from_transactions = transformations.create_labels_from_transactions
transform_enriched_dataset = transformations.transform_enriched_dataset
split_train_val_windows = transformations.split_train_val_windows
save_weekly_features_labels_from_transformed = transformations.save_weekly_features_labels_from_transformed

# -------- DAG DEFINITION --------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
}

with DAG(
    dag_id='etl_features_labels_pipeline_v2',
    default_args=default_args,
    start_date=datetime(2025, 6, 29),
    schedule_interval=None,
    catchup=False,
    tags=['etl', 'features', 'labels']
) as dag:

    def task_unique_pairs(**kwargs):
        path = get_unique_customer_product_pairs()
        kwargs['ti'].xcom_push(key='pairs_path', value=path)

    def task_enrich(**kwargs):
        ti = kwargs['ti']
        pairs_path = ti.xcom_pull(task_ids='unique_pairs', key='pairs_path')
        enriched_path = merge_with_features(pairs_path)
        ti.xcom_push(key='enriched_path', value=enriched_path)

    def task_labels(**kwargs):
        label_path = create_labels_from_transactions()
        kwargs['ti'].xcom_push(key='labels_path', value=label_path)

    def task_transform(**kwargs):
        transformed_path = transform_enriched_dataset()
        kwargs['ti'].xcom_push(key='transformed_path', value=transformed_path)


    def task_save_weekly(**kwargs):
        ti = kwargs['ti']
        transformed_enriched_path = ti.xcom_pull(task_ids='transform_features', key='transformed_path')
        result = save_weekly_features_labels_from_transformed(transformed_enriched_path)
        kwargs['ti'].xcom_push(key='saved_weekly', value=result)

    unique_pairs = PythonOperator(
        task_id='unique_pairs',
        python_callable=task_unique_pairs
    )

    enrich = PythonOperator(
        task_id='enrich_dataset',
        python_callable=task_enrich
    )

    labels = PythonOperator(
        task_id='generate_labels',
        python_callable=task_labels
    )

    transform = PythonOperator(
        task_id='transform_features',
        python_callable=task_transform
    )

    save = PythonOperator(
        task_id='save_weekly_labels',
        python_callable=task_save_weekly
    )

    # Flujo final
    unique_pairs >> enrich
    enrich >> [transform, labels] >> save