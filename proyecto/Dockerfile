FROM python:3.109-slim

ENV DEBIAN_FRONTEND=noninteractive \
    AIRFLOW_HOME=/opt/airflow

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "apache-airflow==2.8.1" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt" 

COPY ./dags ${AIRFLOW_HOME}/dags
COPY ./scripts ${AIRFLOW_HOME}/scripts
COPY ./requirements.txt ${AIRFLOW_HOME}/scripts

RUN pip install -r ${AIRFLOW_HOME}/scripts/requirements.txt

WORKDIR ${AIRFLOW_HOME}

# Sin entrypoint personalizado
CMD ["airflow", "version"]
