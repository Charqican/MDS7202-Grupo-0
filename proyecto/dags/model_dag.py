from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os

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
model = dynamic_import_from_path('model', scripts_path)

# Funciones importadas
IncrementalXGBoost = model.IncrementalXGBoost

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('incremental_xgboost_training', default_args=default_args, schedule_interval='@weekly', catchup=False) as dag:

    def load_or_init_model(**kwargs):
        model = IncrementalXGBoost.load_model()
        # Guardamos modelo en XCom para que otras tareas accedan
        kwargs['ti'].xcom_push(key='model', value=model)

    def generate_predictions(**kwargs):
        ti = kwargs['ti']
        model = ti.xcom_pull(key='model', task_ids='load_or_init_model')
        predictions = model.generate_predictions_for_next_week()
        # Guardamos predicciones y modelo actualizado (por si hay cambios)
        ti.xcom_push(key='model', value=model)
        ti.xcom_push(key='predictions', value=predictions)

    def check_labels_exist(**kwargs):
        ti = kwargs['ti']
        model = ti.xcom_pull(key='model', task_ids='generate_predictions')
        current_week = model.current_week_to_predict
        labels_path = os.path.join('./data/processed/labels/', f"labels_week_{current_week}.parquet")
        if os.path.exists(labels_path):
            return 'evaluate_model'
        else:
            return 'end_no_labels'

    def evaluate_model(**kwargs):
        ti = kwargs['ti']
        model = ti.xcom_pull(key='model', task_ids='generate_predictions')
        f1 = model.evaluate_and_detect_drift()
        # Aquí puedes adaptar evaluate_and_detect_drift para que retorne (f1, needs_retrain)
        # Ejemplo: needs_retrain = True si drop de F1 > umbral
        needs_retrain = False
        if model.last_evaluated_f1 is not None and model.last_evaluated_f1 < (model.f1_threshold_drop):
            needs_retrain = True
        ti.xcom_push(key='model', value=model)
        ti.xcom_push(key='needs_retrain', value=needs_retrain)
        if needs_retrain:
            return 'full_retrain'
        else:
            return 'incremental_train'

    def full_retrain(**kwargs):
        ti = kwargs['ti']
        model = ti.xcom_pull(key='model', task_ids='evaluate_model')
        model.full_retrain()
        ti.xcom_push(key='model', value=model)

    def incremental_train(**kwargs):
        ti = kwargs['ti']
        model = ti.xcom_pull(key='model', task_ids='evaluate_model')
        current_week = model.current_week_to_predict
        model.train_incremental_and_update_history(str(current_week))
        ti.xcom_push(key='model', value=model)

    def save_model(**kwargs):
        ti = kwargs['ti']
        model = ti.xcom_pull(key='model', task_ids=['incremental_train', 'full_retrain'])
        model.save_model()

    
    start = DummyOperator(task_id='start')
    end_no_labels = DummyOperator(task_id='end_no_labels')

    t1 = PythonOperator(task_id='load_or_init_model', python_callable=load_or_init_model, provide_context=True)
    t2 = PythonOperator(task_id='generate_predictions', python_callable=generate_predictions, provide_context=True)
    t3 = PythonOperator(task_id='check_labels_exist', python_callable=check_labels_exist, provide_context=True)
    t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, provide_context=True)
    t5 = PythonOperator(task_id='full_retrain', python_callable=full_retrain, provide_context=True)
    t6 = PythonOperator(task_id='incremental_train', python_callable=incremental_train, provide_context=True)
    t7 = PythonOperator(task_id='save_model', python_callable=save_model, provide_context=True)

    start >> t1 >> t2 >> t3
    t3 >> end_no_labels
    t3 >> t4
    t4 >> [t5, t6] 
    [t5, t6] >> t7