# usamos la imagen oficial p ero instalamos nuestas cosas
FROM apache/airflow:2.8.1-python3.10 


# instalando los paquetes.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt