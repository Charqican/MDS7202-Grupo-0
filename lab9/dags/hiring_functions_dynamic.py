import os
import shutil
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import joblib
import gradio as gr


def create_folders(**kwargs):
    """
    Crea una carpeta con nombre = fecha de ejecución (YYYY-MM-DD) dentro de 'dags',
    copia data_1.csv a raw/ y crea subcarpetas 'splits' y 'models'.
    """
    # 1) Obtener fecha de ejecución
    execution_date = kwargs.get('ds') or datetime.today().strftime('%Y-%m-%d')

    # 2) Directorio 'dags/'
    dags_dir = os.path.dirname(__file__)
    date_folder = os.path.join(dags_dir, execution_date)
    os.makedirs(date_folder, exist_ok=True)

    # 3) Crear raw/ y copiar data_1.csv
    raw_dir = os.path.join(date_folder, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    project_root = os.path.dirname(dags_dir)

    # 4) Crear splits/ y models/
    for sub in ['splits', 'models', 'preprocessed']:
        os.makedirs(os.path.join(date_folder, sub), exist_ok=True)

    return date_folder


def load_ands_merge(**kwargs):
    execution_date = kwargs.get('ds') or datetime.today().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(__file__)
    date_folder = os.path.join(base_dir, execution_date)
    preprocessed_path = os.path.join(date_folder, 'preprocessed')
    print(date_folder)
    raw_path = os.path.join(date_folder, 'raw')

    if len(os.listdir(raw_path)) == 0:
        print('No data')
        return
    dfs = []
    for filename in os.listdir(raw_path):
        full_path = os.path.join(raw_path, filename)
        if os.path.isfile(full_path):
            dfs.append(pd.read_csv(full_path))

    data = pd.concat(dfs, ignore_index=True)
    data.to_csv(os.path.join(preprocessed_path, 'merged_data.csv'), index=False)


def split_data(**kwargs):
    """
    Lee 'dags/<fecha>/raw/data_1.csv', hace hold-out 80/20 y guarda train.csv y test.csv en splits/.
    """
    execution_date = kwargs.get('ds') or datetime.today().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(__file__)
    date_folder = os.path.join(base_dir, execution_date)

    preprocessed_path = os.path.join(date_folder, 'preprocessed', 'merged_data.csv')
    df = pd.read_csv(preprocessed_path)

    X = df.drop('HiringDecision', axis=1)
    y = df['HiringDecision']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=123
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    splits_dir = os.path.join(date_folder, 'splits')
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)


def train_model(model, **kwargs):
    """
    Carga train/test de splits/, construye pipeline (preprocesamiento + RandomForest),
    imprime accuracy y f1, y guarda pipeline.joblib en models/.
    """
    execution_date = kwargs.get('ds') or datetime.today().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(__file__)
    date_folder = os.path.join(base_dir, execution_date)
    splits_dir = os.path.join(date_folder, 'splits')
    models_dir = os.path.join(date_folder, 'models')
    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    X_train = train_df.drop('HiringDecision', axis=1)
    y_train = train_df['HiringDecision']

    numeric_features = [
        'Age','ExperienceYears','PreviousCompanies',
        'DistanceFromCompany','InterviewScore',
        'SkillScore','PersonalityScore'
    ]
    categorical_features = ['Gender','EducationLevel','RecruitmentStrategy']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, f'pipeline_{model.__class__.__name__}.joblib'))


def evaluate_models(**kwargs):
    execution_date = kwargs.get('ds') or datetime.today().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(__file__)
    date_folder = os.path.join(base_dir, execution_date)
    models_path = os.path.join(date_folder, 'models')
    split_path = os.path.join(date_folder, 'splits')
    test_df = pd.read_csv(os.path.join(split_path, 'test.csv'))

    X_test = test_df.drop('HiringDecision', axis=1)
    y_test = test_df['HiringDecision']

    for file in os.listdir(models_path):
        full_path = os.path.join(models_path, file)
        acc_list = []
        if os.path.isfile(full_path):
            pipeline = joblib.load(full_path)

            print(f"Evaluando modelo: {pipeline.named_steps['classifier'].__class__.__name__}")
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            acc_list.append((acc, pipeline))
            print(f"Accuracy en test modelo {pipeline.named_steps['classifier'].__class__.__name__}: {acc:.4f}")
        

        acc_list.sort(key= lambda x: x[0], reverse=True)
        print(f"Mejor modelo: {acc_list[0][1].named_steps['classifier'].__class__.__name__} con accuracy {acc_list[0][0]:.4f}")
        joblib.dump(acc_list[0][1], os.path.join(models_path, 'best_model.joblib'))