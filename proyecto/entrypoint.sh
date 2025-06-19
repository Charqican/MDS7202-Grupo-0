#!/usr/bin/env bash
set -e

# Inicializa la BD
airflow db init

# Crea usuario admin (si ya existe, ignora el error)
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin || true

# Arranca scheduler + webserver
exec airflow scheduler & exec airflow webserver
