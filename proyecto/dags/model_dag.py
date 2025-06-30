from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

from datetime import datetime
import os
import importlib.util

def get_scripts_base_path():
    airflow_home = os.environ.get('AIRFLOW_HOME')
    if airflow_home:
        return os.path.join(airflow_home, 'scripts')
    return os.path.join(os.getcwd(), 'scripts')


def get_data_base_path():
    airflow_home = os.environ.get('AIRFLOW_HOME')
    if airflow_home:
        return os.path.join(airflow_home, 'data')
    return os.path.join(os.getcwd(), 'data')

import sys
sys.path.append('/opt/airflow/scripts')
from IncrementalXGBoost import IncrementalXGBoost

# --- Setup inicial ---
scripts_path = get_scripts_base_path()
MODEL_STATE_PATH = os.path.join(get_data_base_path(),'model_state' )
MODEL_PATH = os.path.join(get_data_base_path(),'model_state/incremental_xgboost_model.pkl' )

default_args = {
    'owner': 'data_team',
    'start_date': datetime(2024, 1, 1),
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    'train_backlog',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['training', 'model', 'xgboost']
) as dag:

    def load_or_create_model(**kwargs):
        ti = kwargs['ti']
        if not os.path.exists(MODEL_PATH):
            model = IncrementalXGBoost()
            model.save_model()
            print(f"Nuevo modelo creado y guardado en {MODEL_PATH}")
        else:
            print(f"Modelo ya existe en {MODEL_PATH}")
        ti.xcom_push(key='model_path', value=MODEL_PATH)

    def conditional_initial_train(**kwargs):
        ti = kwargs['ti']
        model_path = ti.xcom_pull(task_ids='load_or_create_model', key='model_path')
        model = IncrementalXGBoost.load_model()

        if model.model is None:
            print("Modelo no entrenado. Ejecutando entrenamiento inicial...")
            model.initial_train()
            model.save_model()
        else:
            print("Modelo ya entrenado. Saltando entrenamiento inicial.")

    def run_backlog_training(**kwargs):
        ti = kwargs['ti']
        model_path = ti.xcom_pull(task_ids='load_or_create_model', key='model_path')
        model = IncrementalXGBoost.load_model()

        print("Ejecutando entrenamiento backlog hasta la última semana disponible...")
        model.train_backlog_until_latest()
        model.save_model()

    # --- Tareas ---
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    t1 = PythonOperator(
        task_id='load_or_create_model',
        python_callable=load_or_create_model,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id='conditional_initial_train',
        python_callable=conditional_initial_train,
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id='run_backlog_training',
        python_callable=run_backlog_training,
        provide_context=True,
    )

    # --- Flujo ---
    start >> t1 >> t2 >> t3 >> end
