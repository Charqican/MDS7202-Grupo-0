import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import xgboost as xgb
import pickle
import os

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns
from optuna.visualization import plot_optimization_history, plot_param_importances

CURRENT_WORK_DIRECTORY = os.getcwd()
DATA_DIRECTORY_PATH = os.path.join(CURRENT_WORK_DIRECTORY, "data")
PLOT_DIRECTORY = os.path.join(CURRENT_WORK_DIRECTORY, "plots")
MODEL_DIRECTORY = os.path.join(CURRENT_WORK_DIRECTORY, 'models')
DATA_PATH = os.path.join(DATA_DIRECTORY_PATH, 'water_potability.csv')

# 1. Preparar datos
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, preprocessor

# 2. Función Optuna
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42
    }

    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    with mlflow.start_run(run_name=f"XGBoost con lr {params['learning_rate']:.3f}"):
        mlflow.log_params(params)
        mlflow.log_metric("valid_f1", f1)
        mlflow.sklearn.log_model(model, "model")  # ESTA LÍNEA ES CLAVE

    return f1


# 3. Función principal
def optimize_model():
    global X_train, X_test, y_train, y_test, MODEL_DIRECTORY, PLOT_DIRECTORY  # para acceso desde objective()
    X_train, X_test, y_train, y_test, preprocessor = load_data()

    # Crear carpetas si no existen
    os.makedirs(PLOT_DIRECTORY, exist_ok=True)
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

    experiment_name = "XGBoost Water Potability Optuna"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # Guardar gráficos de Optuna
    fig1 = plot_optimization_history(study)
    fig1.write_image(os.path.join(PLOT_DIRECTORY, 'optimization_history.png'))

    fig2 = plot_param_importances(study)
    fig2.write_image(os.path.join(PLOT_DIRECTORY, 'param_importances.png'))

    # Obtener mejor modelo
    best_run_id = mlflow.search_runs(experiment.experiment_id).sort_values("metrics.valid_f1", ascending=False).iloc[0]["run_id"]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    # Guardar modelo con pickle
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Feature importances
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        features = preprocessor.get_feature_names_out()
        importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
        plt.figure(figsize=(10,6))
        sns.barplot(data=importance_df.sort_values("Importance", ascending=False), x="Importance", y="Feature")
        plt.title("Importancia de variables")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIRECTORY, 'feature_importances.png'))

# Esta función permite ejecutar optimize_model directamente con: python optimize.py
if __name__ == "__main__":
    optimize_model()
