FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY frontend .

ENV BACKEND_URL=http://backend:8000

EXPOSE 7860

CMD ["python", "app.py"]
