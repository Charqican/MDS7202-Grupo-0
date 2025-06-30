from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

from datetime import datetime
import os


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

# --- Setup inicial ---
scripts_path = get_scripts_base_path()
model_module = dynamic_import_from_path('model', scripts_path)
IncrementalXGBoost = model_module.IncrementalXGBoost
MODEL_PATH = "opt/home/data/model_state/incremental_xgboost_model.pkl"


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
            model.save_model(MODEL_PATH)
            print(f"Nuevo modelo creado y guardado en {MODEL_PATH}")
        else:
            print(f"Modelo ya existe en {MODEL_PATH}")
        ti.xcom_push(key='model_path', value=MODEL_PATH)

    def conditional_initial_train(**kwargs):
        ti = kwargs['ti']
        model_path = ti.xcom_pull(task_ids='load_or_create_model', key='model_path')
        model = IncrementalXGBoost.load_model(model_path)

        if model.model is None:
            print("Modelo no entrenado. Ejecutando entrenamiento inicial...")
            model.initial_train()
            model.save_model(model_path)
        else:
            print("Modelo ya entrenado. Saltando entrenamiento inicial.")

    def run_backlog_training(**kwargs):
        ti = kwargs['ti']
        model_path = ti.xcom_pull(task_ids='load_or_create_model', key='model_path')
        model = IncrementalXGBoost.load_model(model_path)

        print("Ejecutando entrenamiento backlog hasta la última semana disponible...")
        model.train_backlog_until_latest()
        model.save_model(model_path)

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
