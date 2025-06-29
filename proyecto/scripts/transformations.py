import pandas as pd
import os
import sys
import importlib
import logging

#### IMPORTS 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ------ FUNCION PARA IMPORTACION ------
def dynamic_import_from_path(module_name: str, base_path: str):
    file_path = os.path.join(base_path, f"{module_name}.py")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file not found at: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for {module_name} at {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # intentaremos obtener el modulo
    try:
        spec.loader.exec_module(module)
        logger.info(f"Module '{module_name}' loaded from {file_path}")
    except Exception as e:
        logger.critical(f"Error executing module '{module_name}' from {file_path}: {e}")
        raise ImportError(f"Failed to execute module {module_name} from {file_path}") from e
    
    return module


# ------ FUNCIÓN PARA OBTENER PATHS ------
def get_scripts_base_path():
    airflow_home = os.environ.get('AIRFLOW_HOME')
    
    if airflow_home:
        scripts_path = os.path.join(airflow_home, 'scripts')
        logger.info(f"Airflow environment detected. Scripts path: {scripts_path}")
    else:
        scripts_path = os.path.join(os.getcwd(), 'scripts')
        logger.info(f"Local environment detected. Attempting scripts path: {scripts_path}")
        
        if not os.path.isdir(scripts_path):
            logger.warning(f"Local scripts path '{scripts_path}' does not exist or is not a directory.")
            
    return scripts_path

try:
    scripts_base_path = get_scripts_base_path()
    pipeline_module = dynamic_import_from_path('pipeline', scripts_base_path)

    pipeline_fe = pipeline_module.pipeline_fe
    
    logger.info("All functions from pipeline.py loaded.")

except (FileNotFoundError, ImportError) as e:
    logger.critical(f"FATAL ERROR: Could not import pipeline.py: {e}")
    sys.exit(1)


def get_dirs() -> tuple[str, str, str]:
    """
    Determina rutas raw y processed según AIRFLOW_HOME.
    Retorna las rutas de data/raw, data/processed, data/week_data 
    """
    # Preferencia: usar AIRFLOW_HOME si está disponible, sino './'
    airflow_home = os.environ.get('AIRFLOW_HOME', './')
    raw_dir = os.path.join(airflow_home, 'data', 'raw')
    processed_dir = os.path.join(airflow_home, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    week_data = os.path.join(airflow_home, 'data', 'week_data')
    os.makedirs(week_data, exist_ok=True)
    return raw_dir, processed_dir, week_data


def extract_and_merge() -> str:
    raw_dir, processed_dir, _ = get_dirs()

    df_trans = pd.read_parquet(os.path.join(raw_dir, 'transacciones.parquet'))
    df_cli = pd.read_parquet(os.path.join(raw_dir, 'clientes.parquet'))
    df_prod = pd.read_parquet(os.path.join(raw_dir, 'productos.parquet'))

    df_trans_clean = df_trans.drop_duplicates().copy()
    df_cli_clean = df_cli.drop_duplicates().copy()
    df_prod_clean = df_prod.drop_duplicates().copy()

    df_trans_clean["customer_id"] = df_trans_clean["customer_id"].astype(str)
    df_trans_clean["product_id"] = df_trans_clean["product_id"].astype(str)
    df_cli_clean["customer_id"] = df_cli_clean["customer_id"].astype(str)
    df_prod_clean["product_id"] = df_prod_clean["product_id"].astype(str)

    df_trans_clean["purchase_date"] = pd.to_datetime(df_trans_clean["purchase_date"])

    df_merged = (
        df_trans_clean
        .merge(df_cli_clean, on="customer_id", how="left")
        .merge(df_prod_clean, on="product_id", how="left")
    )

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'merged.parquet')
    df_merged.to_parquet(out_path, index=False)
    logger.info(f"Saved merged data at {out_path}")

    return out_path


def create_label(merge_path: str) -> str:
    """
    Genera etiquetas semanales (0/1) a partir de datos fusionados y limpios.
    Guarda labeled.parquet.

    Args:
        merge_path (str): Ruta al archivo Parquet combinado de entrada (merged.parquet).

    Returns:
        str: La ruta donde se guardó el archivo Parquet etiquetado.
    """
    _, processed_dir, _ = get_dirs()
    df_merged = pd.read_parquet(merge_path)

    # Asegurarse de que purchase_date sea datetime, aunque extract_and_merge ya lo hace
    df_merged['purchase_date'] = pd.to_datetime(df_merged['purchase_date'])

    # Calcular 'year_week'
    df_merged['week_num'] = df_merged['purchase_date'].dt.to_period('W').astype(str)

    # Sumar ítems por semana
    df_semanal = (
        df_merged
        .groupby(['customer_id', 'product_id', 'week_num'], as_index=False)
        .agg(items_sum=('items', 'sum'))
    )

    # Obtener pares únicos (customer_id, product_id)
    pares_unicos = df_merged[['customer_id', 'product_id']].drop_duplicates()

    # Obtener todas las semanas completas en el rango de fechas
    fechas_completas = pd.date_range(df_merged['purchase_date'].min(), df_merged['purchase_date'].max(), freq='D')
    semanas_completas = pd.Series(fechas_completas.to_period('W').unique().astype(str), name='week_num')
    
    # Mapeo de 'YYYYWXX' a 'week_N'
    semanas_completas_map = {week : f'week_{i+1}' for i, week in enumerate(semanas_completas)}

    # Crear todas las combinaciones válidas de pares y semanas
    # Este enfoque es más robusto que MultiIndex.from_product directo si hay IDs con diferente rango de fechas
    combinaciones_validas = pd.MultiIndex.from_frame(
        pares_unicos.assign(key=1).merge(semanas_completas.to_frame().assign(key=1), on='key').drop('key', axis=1)
    )

    # Reindexar para asegurar todas las combinaciones posibles y rellenar con 0
    df_semanal_idx = df_semanal.set_index(['customer_id', 'product_id', 'week_num'])
    df_full = df_semanal_idx.reindex(combinaciones_validas, fill_value=0).reset_index()
    
    # Generar la etiqueta (1 si items_sum > 0, 0 en caso contrario)
    df_full['label'] = (df_full['items_sum'] > 0).astype(int)
    
    # Renombrar 'year_week' a 'week_N'
    df_full['week_num'] = df_full['week_num'].map(semanas_completas_map)
    
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'labeled.parquet')
    df_full.to_parquet(out_path, index=False)
    
    return out_path


def prepare_model_dataset(merge_path: str, labeled_path: str) -> str:
    """
    Construye dataset con atributos constantes y filtra tras primera compra.
    Guarda un dataset filtrado y retorna su directorio.
    """
    _, processed_dir,_ = get_dirs()
    df_merged = pd.read_parquet(merge_path)
    df_label  = pd.read_parquet(labeled_path)
    df_client = (
        df_merged[['customer_id','region_id','zone_id','customer_type',
                   'Y','X','num_deliver_per_week','num_visit_per_week']]
        .drop_duplicates(subset='customer_id') 
    )
    df_prod = (
        df_merged[['product_id','brand','category','sub_category','segment','package','size']]
        .drop_duplicates(subset='product_id') 
    )

    df_model = (
        df_label
        .merge(df_client, on='customer_id', how='left')
        .merge(df_prod, on='product_id', how='left')
    )


    df_model = df_model.drop(columns=[c for c in df_model.columns if 'purchase_date' in c], errors='ignore')


    df_model['week_num'] = df_model['week_num'].astype(str).str.replace('week_','').astype(int)
    
    def filter_fun(group):
        purchase = group[group['label'] == 1]
        if purchase.empty:
            return pd.DataFrame() # Si no hay compras, no hay registros para mantener
        primera_semana = purchase['week_num'].min()
        return group[(group['label'] == 1) | (group['week_num'] >= primera_semana)]

    df_modelo_filtrado = df_model.groupby(['customer_id', 'product_id'], group_keys=False).apply(filter_fun)
    df_modelo_filtrado['week_num'] = df_modelo_filtrado['week_num'].astype(int)

    filtrado_path = os.path.join(processed_dir,'filtrado.parquet')
    df_modelo_filtrado.to_parquet(filtrado_path)
    
    return filtrado_path


def transform_features(df: pd.DataFrame) -> str:
    _, processed_dir,_ = get_dirs()
    drop_zero_var = ['region_id', 'zone_id', 'num_visit_per_week', 'items_sum']

    def filter_xy(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df['X'] > -108) & (df['X'] < -107) & (df['Y'] > -48) & (df['Y'] < -46)]

    df_filtered = filter_xy(df.copy()).drop(columns=drop_zero_var, errors='ignore')
    transformed = pipeline_fe.fit_transform(df_filtered)
    df_transformed = pd.DataFrame(transformed, columns=pipeline_fe.get_feature_names_out())
    transformado_path = os.path.join(processed_dir,'transformed.parquet')
    df_transformed.to_parquet(transformado_path)
    return transformado_path


def split_train_val_windows(dataset_path: str) -> list[str]:
    _, _, week_data_dir = get_dirs()
    os.makedirs(week_data_dir, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    if 'week_num' not in df.columns:
        raise KeyError("La columna 'week_num' no existe en el dataset de features")

    paths = []
    for week in sorted(df['week_num'].unique()):
        df_week = df[df['week_num'] == week]
        out_file = os.path.join(week_data_dir, f'week_{int(week)}.parquet')
        df_week.to_parquet(out_file, index=False)
        paths.append(out_file)

    logger.info(f"Split dataset into {len(paths)} weekly files in {week_data_dir}")
    return paths


def save_weekly_features_labels(weekly_files: list[str]) -> dict[str, list[str]]:
    _, processed_dir, _ = get_dirs()
    features_dir = os.path.join(processed_dir, 'features')
    labels_dir = os.path.join(processed_dir, 'labels')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    feat_paths = []
    label_paths = []

    for week_file in sorted(weekly_files):
        df_week = pd.read_parquet(week_file)

        base_name = os.path.basename(week_file)
        week_num = base_name.replace('week_', '').replace('.parquet', '')

        feat_df = df_week.drop(columns=['label'], errors='ignore')
        feat_file = os.path.join(features_dir, f'week_{int(week_num)}.parquet')
        feat_df.to_parquet(feat_file, index=False)
        feat_paths.append(feat_file)

        lbl_df = df_week[['label']]
        lbl_file = os.path.join(labels_dir, f'label_week_{int(week_num)}.parquet')
        lbl_df.to_parquet(lbl_file, index=False)
        label_paths.append(lbl_file)

    logger.info(f"Saved features and labels separately for {len(weekly_files)} weeks")
    return {'features': feat_paths, 'labels': label_paths}