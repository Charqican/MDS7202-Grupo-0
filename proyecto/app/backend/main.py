from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc  # si usas MLflow, o pickle si usas joblib/pickle

# Define el contrato de entrada
class InputData(BaseModel):
    customer_id: int
    product_id: int
    year_week: str
    X: float
    Y: float
    size: float
    num_deliver_per_week: int
    # …otras features que tu modelo requiere

app = FastAPI(title="SodAI Drinks Predictor")

# Al arrancar, carga el modelo en memoria
model = mlflow.pyfunc.load_model("models:/sodai_drinks/Production")  # ejemplo MLflow
# O bien:
# import joblib
# model = joblib.load("/opt/models/mimodelo.pkl")

@app.post("/predict/")
def predict(data: InputData):
    # Convierte el payload a DataFrame de 1 fila
    df = pd.DataFrame([data.dict()])
    # Aquí podrías reusar tu transform_features
    from scripts.transformations import transform_features
    df_feat = transform_features(df)
    # Obtiene la predicción
    pred = model.predict(df_feat)
    return {"prediction": float(pred[0])}
