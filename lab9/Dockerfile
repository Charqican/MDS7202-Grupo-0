FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

ENV AIRFLOW_HOME=/opt/airflow
RUN pip install apache-airflow==2.8.1 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt"
RUN pip install pandas scikit-learn gradio joblib


RUN mkdir -p ${AIRFLOW_HOME}/dags
COPY dags/ ${AIRFLOW_HOME}/dags/

WORKDIR ${AIRFLOW_HOME}

RUN airflow db init
RUN airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
CMD ["sh","-c","airflow webserver & airflow scheduler"]
