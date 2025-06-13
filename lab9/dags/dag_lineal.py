from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

# Importar funciones auxiliares
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# Definición del DAG
with DAG(
    dag_id='hiring_lineal',                
    start_date=datetime(2024, 10, 1),      
    schedule_interval=None,                
    catchup=False,                         
    tags=['hiring', 'lineal']              
) as dag:

    # 1) Marcador de inicio del pipeline
    start_pipeline = DummyOperator(
        task_id='start_pipeline'
    )

    # 2) Crear carpetas y estructura (raw, splits, models)
    create_dirs = PythonOperator(
        task_id='create_folders',
        python_callable=create_folders,
        op_kwargs={'ds': '{{ ds }}'}
    )

    # 3) Descargar data_1.csv al directorio raw de la ejecución actual
    download_data = BashOperator(
        task_id='download_data',
        bash_command=(
            "curl -sSL -o "
            "{{ ti.xcom_pull(task_ids='create_folders') }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        )
    )

    # 4) Aplicar hold-out
    split = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        op_kwargs={'ds': '{{ ds }}'}
    )

    # 5) Preprocesar y entrenar
    train = PythonOperator(
        task_id='preprocess_and_train',
        python_callable=preprocess_and_train,
        op_kwargs={'ds': '{{ ds }}'}
    )

    # 6) Interfaz Gradio
    gradio_app = PythonOperator(
        task_id='gradio_interface',
        python_callable=gradio_interface
    )

    # Flujo
    start_pipeline >> create_dirs >> download_data >> split >> train >> gradio_app
