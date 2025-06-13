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
    #src = os.path.join(project_root, 'data_1.csv')
    #dst = os.path.join(raw_dir, 'data_1.csv')
    #if not os.path.exists(src):
    #    raise FileNotFoundError(f"No encuentro data_1.csv en {src}")
    #shutil.copy(src, dst)

    # 4) Crear splits/ y models/
    for sub in ['splits', 'models']:
        os.makedirs(os.path.join(date_folder, sub), exist_ok=True)

    return date_folder


def split_data(**kwargs):
    """
    Lee 'dags/<fecha>/raw/data_1.csv', hace hold-out 80/20 y guarda train.csv y test.csv en splits/.
    """
    execution_date = kwargs.get('ds') or datetime.today().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(__file__)
    date_folder = os.path.join(base_dir, execution_date)

    raw_path = os.path.join(date_folder, 'raw', 'data_1.csv')
    df = pd.read_csv(raw_path)

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


def preprocess_and_train(**kwargs):
    """
    Carga train/test de splits/, construye pipeline (preprocesamiento + RandomForest),
    imprime accuracy y f1, y guarda pipeline.joblib en models/.
    """
    execution_date = kwargs.get('ds') or datetime.today().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(__file__)
    date_folder = os.path.join(base_dir, execution_date)
    splits_dir = os.path.join(date_folder, 'splits')

    train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))

    X_train = train_df.drop('HiringDecision', axis=1)
    y_train = train_df['HiringDecision']
    X_test = test_df.drop('HiringDecision', axis=1)
    y_test = test_df['HiringDecision']

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
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy en test: {acc:.4f}")
    print(f"F1-score (positiva=1): {f1:.4f}")

    models_dir = os.path.join(date_folder, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, 'pipeline.joblib'))


def gradio_interface():
    """
    Lanza interfaz Gradio cargando el pipeline desde models/ de la fecha actual.
    """
    base_dir = os.path.dirname(__file__)
    today_str = datetime.today().strftime('%Y-%m-%d')
    model_path = os.path.join(base_dir, today_str, 'models', 'pipeline.joblib')
    model = joblib.load(model_path)

    def predict(
        age, gender, education, exp_years, prev_companies,
        distance, interview_score, skill_score, personality_score,
        strategy
    ):
        input_df = pd.DataFrame([{
            'Age': age,'Gender': gender,'EducationLevel': education,
            'ExperienceYears': exp_years,'PreviousCompanies': prev_companies,
            'DistanceFromCompany': distance,'InterviewScore': interview_score,
            'SkillScore': skill_score,'PersonalityScore': personality_score,
            'RecruitmentStrategy': strategy
        }])
        pred = model.predict(input_df)[0]
        return 'Contratado' if pred == 1 else 'No Contratado'

    with gr.Blocks() as demo:
        gr.Markdown("## Predicción de decisión de contratación")
        age = gr.Number(label='Edad', value=30)
        gender = gr.Dropdown([0,1], label='Género (0=Male,1=Female)', value=0)
        education = gr.Dropdown([1,2,3,4], label='Nivel Educativo', value=2)
        exp_years = gr.Number(label='Años experiencia', value=5)
        prev_companies = gr.Number(label='Núm compañías previas', value=2)
        distance = gr.Number(label='Distancia (km)', value=10)
        interview_score = gr.Number(label='Puntaje entrevista', value=80)
        skill_score = gr.Number(label='Puntaje habilidades', value=75)
        personality_score = gr.Number(label='Puntaje personalidad', value=70)
        strategy = gr.Dropdown([1,2,3], label='Estrategia reclutamiento', value=2)
        output = gr.Textbox(label='Predicción')
        submit = gr.Button('Predecir')
        submit.click(
            fn=predict,
            inputs=[age,gender,education,exp_years,prev_companies,
                    distance,interview_score,skill_score,personality_score,
                    strategy],
            outputs=output
        )
    demo.launch(share=True)


if __name__ == "__main__":
    # Simula ejecución desde CMD
    today = datetime.today().strftime("%Y-%m-%d")

    folder = create_folders(ds=today)
    print(f"✅ Carpetas creadas en: {folder}")

    split_data(ds=today)
    print("✅ Split train/test completado.")

    preprocess_and_train(ds=today)
    print("✅ Preprocesamiento y entrenamiento finalizados.")

    gradio_interface()
