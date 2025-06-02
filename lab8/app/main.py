from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os


CURRENT_WORK_DIRECTORY = os.getcwd()
DATA_DIRECTORY_PATH = os.path.join(CURRENT_WORK_DIRECTORY, "data")
PLOT_DIRECTORY = os.path.join(CURRENT_WORK_DIRECTORY, "plots")
MODEL_DIRECTORY = os.path.join(CURRENT_WORK_DIRECTORY, 'models')
DATA_PATH = os.path.join(DATA_DIRECTORY_PATH, 'water_potability.csv')

# Cargar el modelo
with open(os.path.join(MODEL_DIRECTORY, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

# (Opcional: si guardaste el preprocesador, puedes cargarlo igual)
# with open("models/preprocessor.pkl", "rb") as f:
#     preprocessor = pickle.load(f)

app = FastAPI(title="API de Potabilidad de Agua")

# Esquema de entrada con Pydantic
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def home():
    return {
        "mensaje": "Modelo de clasificación binaria para determinar si el agua es potable o no.",
        "entrada": "Valores fisicoquímicos del agua (pH, sólidos, etc.)",
        "salida": "potabilidad = 0 (no potable) o 1 (potable)"
    }

@app.post("/potabilidad/")
def predecir_potabilidad(sample: WaterSample):
    # Convertir la entrada en array 2D
    data = np.array([[sample.ph, sample.Hardness, sample.Solids, sample.Chloramines,
                      sample.Sulfate, sample.Conductivity, sample.Organic_carbon,
                      sample.Trihalomethanes, sample.Turbidity]])
    
    # Suponemos que los datos ya están preprocesados; si no, aplicar preprocesamiento aquí
    pred = model.predict(data)[0]
    
    return {"potabilidad": int(pred)}
