#!/usr/bin/env bash
set -e

# Inicializa DB si no existe
if [ ! -f "${AIRFLOW_HOME}/airflow.db" ]; then
    echo "===> airflow.db no existe, inicializando base de datos..."
    airflow db init
else
    echo "===> airflow.db ya existe, omitiendo inicialización"
fi

# Crea usuario admin (si ya existe, saltar)
if ! airflow users list | grep -q 'admin'; then
  airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
fi

# Arranca scheduler + webserver
exec airflow scheduler & exec airflow webserver
