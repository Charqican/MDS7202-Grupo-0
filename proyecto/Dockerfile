# Dockerfile
# Dockerfile
FROM python:3.10-slim

# evita prompts y define AIRFLOW_HOME
ENV DEBIAN_FRONTEND=noninteractive \
    AIRFLOW_HOME=/opt/airflow

# Dependencias de SO
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl build-essential default-libmysqlclient-dev \
 && rm -rf /var/lib/apt/lists/*

# Instala Airflow + pandas
RUN pip install \
      "apache-airflow==2.8.1" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt" \
  && pip install pandas pyarrow fastparquet

# crea usuario airflow
RUN groupadd -g 50000 airflow \
 && useradd -u 50000 -g airflow -m -d /home/airflow -s /bin/bash airflow

USER airflow
WORKDIR ${AIRFLOW_HOME}

# copia DAGs, scripts y el entrypoint
COPY --chown=airflow:airflow dags/    ${AIRFLOW_HOME}/dags/
COPY --chown=airflow:airflow scripts/ ${AIRFLOW_HOME}/scripts/
COPY --chown=airflow:airflow entrypoint.sh ${AIRFLOW_HOME}/entrypoint.sh
RUN sed -i 's/\r$//' ${AIRFLOW_HOME}/entrypoint.sh \
    && chmod +x ${AIRFLOW_HOME}/entrypoint.sh
ENTRYPOINT ["bash", "/opt/airflow/entrypoint.sh"]
