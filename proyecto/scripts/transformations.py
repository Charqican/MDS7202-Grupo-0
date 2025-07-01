import pandas as pd
import os
import sys
import importlib
import logging
import coloredlogs
#### IMPORTS 

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

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



def get_unique_customer_product_pairs() -> str:
    """
    Genera el producto cartesiano entre todos los customer_id y product_id únicos.
    Guarda como unique_pairs.parquet.

    Returns:
        str: Ruta al archivo guardado.
    """
    raw_dir, processed_dir, _ = get_dirs()
    trans_path = os.path.join(raw_dir, 'transacciones.parquet')
    
    logger.info(f"Loading transactions from {trans_path}")
    df_trans = pd.read_parquet(trans_path)
    df_trans["customer_id"] = df_trans["customer_id"].astype(str)
    df_trans["product_id"] = df_trans["product_id"].astype(str)

    unique_customers = df_trans[['customer_id']].drop_duplicates()
    unique_products = df_trans[['product_id']].drop_duplicates()

    logger.info(f"Found {len(unique_customers)} unique customers and {len(unique_products)} unique products.")

    # Crear producto cartesiano usando merge con clave auxiliar
    unique_customers['key'] = 1
    unique_products['key'] = 1
    unique_pairs = unique_customers.merge(unique_products, on='key').drop('key', axis=1)

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'unique_pairs.parquet')
    unique_pairs.to_parquet(out_path, index=False)

    logger.info(f"Saved {len(unique_pairs)} customer-product pairs (cartesian product) to {out_path}")
    return out_path



def merge_with_features(pairs_path: str) -> str:
    """
    Realiza un merge del dataset de pares únicos con información de clientes y productos.

    Args:
        pairs_path (str): Ruta al archivo con los pares únicos.

    Returns:
        str: Ruta del archivo enriquecido.
    """
    raw_dir, processed_dir, _ = get_dirs()

    logger.info(f"Loading unique pairs from {pairs_path}")
    df_pairs = pd.read_parquet(pairs_path)
    df_cli = pd.read_parquet(os.path.join(raw_dir, 'clientes.parquet'))
    df_prod = pd.read_parquet(os.path.join(raw_dir, 'productos.parquet'))

    df_cli["customer_id"] = df_cli["customer_id"].astype(str)
    df_prod["product_id"] = df_prod["product_id"].astype(str)

    df_enriched = (
        df_pairs
        .merge(df_cli, on='customer_id', how='left')
        .merge(df_prod, on='product_id', how='left')
    )

    out_path = os.path.join(processed_dir, 'enriched_pairs.parquet')
    df_enriched.to_parquet(out_path, index=False)

    logger.info(f"Saved enriched pairs with shape {df_enriched.shape} to {out_path}")
    return out_path


def create_labels_from_transactions() -> str:
    """
    Genera etiquetas semanales (0/1) a partir de transacciones históricas.
    Solo incluye combinaciones (cliente, producto, semana) que hayan ocurrido.
    
    Returns:
        str: Ruta al archivo Parquet con las etiquetas generadas.
    """
    raw_dir, processed_dir, _ = get_dirs()
    trans_path = os.path.join(raw_dir, 'transacciones.parquet')
    logger.info(f"Loading transactions from {trans_path}")
    
    df_trans = pd.read_parquet(trans_path)

    # Formateo de columnas
    df_trans['customer_id'] = df_trans['customer_id'].astype(str)
    df_trans['product_id'] = df_trans['product_id'].astype(str)
    df_trans['purchase_date'] = pd.to_datetime(df_trans['purchase_date'])

    # Calcular semana calendario
    base_date = pd.Timestamp('2024-01-01')
    df_trans['week_num'] = ((df_trans['purchase_date'] - base_date).dt.days // 7 + 1).astype(int)
    logger.info(f"Transactions span {df_trans['week_num'].nunique()} unique weeks.")

    # Agrupar para obtener la etiqueta
    df_labels = (
        df_trans
        .groupby(['customer_id', 'product_id', 'week_num'], as_index=False)
        .agg(items_sum=('items', 'sum'))
    )
    
    df_labels['label'] = (df_labels['items_sum'] > 0).astype(int)
    df_labels.drop(columns=['items_sum'], inplace=True)

    out_path = os.path.join(processed_dir, 'weekly_labels.parquet')
    os.makedirs(processed_dir, exist_ok=True)
    df_labels.to_parquet(out_path, index=False)

    logger.info(f"Weekly labels dataset saved to {out_path} with {len(df_labels)} rows.")
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


def transform_enriched_dataset() -> str:
    """
    Aplica transformaciones sobre el dataset enriquecido de clientes-productos.
    Guarda el resultado como archivo Parquet.

    Returns:
        str: Ruta al archivo transformado.
    """
    _, processed_dir, _ = get_dirs()
    enriched_path = os.path.join(processed_dir, 'enriched_pairs.parquet')
    
    logger.info(f"Loading enriched dataset from {enriched_path}")
    df = pd.read_parquet(enriched_path)

    drop_zero_var = ['region_id', 'zone_id', 'num_visit_per_week', 'items_sum']

    def filter_xy(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df['X'] > -108) & (df['X'] < -107) & (df['Y'] > -48) & (df['Y'] < -46)]

    df_filtered = filter_xy(df.copy()).drop(columns=drop_zero_var, errors='ignore')

    logger.info(f"Applying feature transformation pipeline on {len(df_filtered)} records.")
    transformed = pipeline_fe.fit_transform(df_filtered)
    df_transformed = pd.DataFrame(transformed, columns=pipeline_fe.get_feature_names_out())

    # 👇 Añadir customer_id y product_id desde df_filtered
    df_transformed.insert(0, 'customer_id', df_filtered['customer_id'].values)
    df_transformed.insert(1, 'product_id', df_filtered['product_id'].values)

    transformado_path = os.path.join(processed_dir, 'transformed_enriched.parquet')
    df_transformed.to_parquet(transformado_path, index=False)
    
    logger.info(f"Saved transformed enriched dataset to {transformado_path}")
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


def save_weekly_features_labels_from_transformed(transformed_path: str) -> list[str]:
    _, processed_dir, _ = get_dirs()
    labels_path = os.path.join(processed_dir, 'weekly_labels.parquet')
    labels_dir = os.path.join(processed_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    logger.info(f"Loading transformed features from {transformed_path}")
    df_transformed = pd.read_parquet(transformed_path)

    logger.info(f"Loading weekly labels from {labels_path}")
    df_labels = pd.read_parquet(labels_path)

    # Extraer todas las semanas únicas
    all_weeks = sorted(df_labels['week_num'].unique())

    saved_paths = []

    for week in all_weeks:
        # Subset de etiquetas para la semana
        df_week_labels = df_labels[df_labels['week_num'] == week][['customer_id', 'product_id', 'label']]

        # Obtener todas las combinaciones para esa semana, replicando df_transformed pero con la columna week_num fija
        df_week_full = df_transformed.copy()
        df_week_full['week_num'] = week

        # Merge con etiquetas: si no hay etiqueta, es 0
        merged = df_week_full.merge(df_week_labels, on=['customer_id', 'product_id'], how='left')
        merged['label'] = merged['label'].fillna(0).astype(int)

        # Guardar solo columnas clave + label
        cols_to_save = ['week_num', 'label']
        merged = merged[cols_to_save]

        out_path = os.path.join(labels_dir, f'week_labels_{week}.parquet')
        merged.to_parquet(out_path, index=False)
        saved_paths.append(out_path)

    logger.info(f"Saved weekly features+labels datasets for {len(all_weeks)} weeks to {labels_dir}")
    return saved_paths


def merge_transactions():
    """
        Merge new transactions data into the raw transactions file
    """
    airflow_home = os.environ.get('AIRFLOW_HOME', './')
    transactions_data = os.path.join(airflow_home, 'data', 'transactions')
    raw = os.path.join(airflow_home, 'data', 'raw')
    os.makedirs(transactions_data, exist_ok=True)
    transactions_raw_path = os.path.join(raw, 'transactions.parquet')
    transactions_raw = pd.read_parquet(transactions_raw_path)
    week_dataframes = []
    for file in os.listdir(transactions_data):
        file_path = os.path.join(transactions_data, file)
        if os.path.isfile(file_path):
            week_dataframes.append(pd.read_parquet(file_path))
    
    if len(week_dataframes) != 0:
        transactions_raw = pd.concat([transactions_raw]+week_dataframes)
        transactions_raw.to_parquet(transactions_raw_path)
            
            