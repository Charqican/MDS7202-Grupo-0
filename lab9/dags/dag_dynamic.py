from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import os
import logging

# Importar funciones auxiliares

from hiring_functions_dynamic import create_folders, load_ands_merge, split_data, train_model, evaluate_models

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# función auxiliar para decidir la rama de descarga
def branch_download(**kwargs):
    execution_date_str = kwargs.get('ds')
    execution_date = datetime.strptime(execution_date_str, "%Y-%m-%d") 
    if execution_date < datetime(2024, 11, 1):
        return 'download_data_1'
    else:
        return 'download_both'



# Get the Airflow task logger
log = logging.getLogger(__name__)

def check_data_files(**kwargs):

    """
    Check if data files exist in the raw directory.
    If no files are found, skip the processing step.
    """
    execution_date = kwargs.get('ds')
    base_dir = os.path.dirname(__file__)
    date_folder = os.path.join(base_dir, execution_date)
    raw_path = os.path.join(date_folder, 'raw')

    if not os.listdir(raw_path):
        log.info("No data files found in raw directory. Skipping processing.")
        return 'skip_processing'
    
    log.info("Data files found. Proceeding with processing.")
    return 'merge_data'

with DAG(dag_id='hiring_dynamic', start_date=datetime(2024, 10, 1), schedule_interval=timedelta(days=5), catchup=False, tags=['hiring', 'lineal']) as dag:
    
    start_pipeline = DummyOperator(
        task_id='start_pipeline'
    )

    create_dirs = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'ds': '{{ ds }}'}
    )

    choose_branch = BranchPythonOperator(
        task_id='branching_download',
        python_callable=branch_download,
        op_kwargs={'ds': '{{ ds }}'},
        provide_context=True
    )

    download_data_1 = BashOperator(
        task_id='download_data_1',
        bash_command=(
            "curl -sSL -o {{ ti.xcom_pull(task_ids='create_folders') }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        )
    )


    download_both = BashOperator(
        task_id='download_both',
        bash_command=(
            "curl -sSL -o {{ ti.xcom_pull(task_ids='create_folders') }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv && "
            "curl -sSL -o {{ ti.xcom_pull(task_ids='create_folders') }}/raw/data_2.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
        )
    )

    check_data_trigger = BranchPythonOperator(
        task_id='check_data_available',
        python_callable=check_data_files,
        provide_context=True,
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    merge = PythonOperator(
        task_id='merge_data',
        python_callable=load_ands_merge,
        op_kwargs={'ds': '{{ ds }}'}
    )

    split = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'ds': '{{ ds }}'}
    )


    train_rf = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_model,
        op_kwargs={
            'model': RandomForestClassifier(),
            'ds': '{{ ds }}'
        }
    )


    train_gb = PythonOperator(
        task_id='train_gb',
        python_callable=train_model,
        op_kwargs={
            'model': GradientBoostingClassifier(),
            'ds': '{{ ds }}'
        }
    )

    train_lr = PythonOperator(
        task_id='train_lr',
        python_callable=train_model,
        op_kwargs={
            'model': LogisticRegression(max_iter=500),
            'ds': '{{ ds }}'
        }
    )

    evaluate = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        op_kwargs={'ds': '{{ ds }}'},
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    skip_processing = DummyOperator(
        task_id='skip_processing'
    )

    end_pipeline = DummyOperator(task_id='end_pipeline', trigger_rule=TriggerRule.ONE_SUCCESS)


    # Estructura del flujo
    start_pipeline >> create_dirs >> choose_branch
    choose_branch >> [download_data_1, download_both ] >> check_data_trigger

    # Branching logic for check_data_trigger
    check_data_trigger >> [merge, skip_processing] # Branch to merge or skip

    # If data is available, proceed with processing
    merge >> split >> [train_rf, train_gb, train_lr] >> evaluate

    # All paths eventually lead to end_pipeline
    [evaluate, skip_processing] >> end_pipeline