import pandas as pd
import os


def get_dirs() -> tuple[str, str, str]:
    """
    Determina rutas raw y processed según AIRFLOW_HOME.
    Retorna las rutas de data/raw, data/processed, data/week_data 
    """
    # Preferencia: usar AIRFLOW_HOME si está disponible, sino './'
    airflow_home = os.environ.get('AIRFLOW_HOME', './')
    raw_dir = os.path.join(airflow_home, 'data', 'raw')
    processed_dir = os.path.join(airflow_home, 'data', 'processed')
    week_data = os.path.join(airflow_home, 'data', 'week_data')
    return raw_dir, processed_dir, week_data


def extract_and_merge() -> str:
    """
    Lee Parquet de transacciones, clientes y productos, los limpia,
    fusiona y guarda merged.parquet en processed.
    """
    raw_dir, processed_dir, _ = get_dirs()

    # 1) Leer DataFrames
    df_trans = pd.read_parquet(os.path.join(raw_dir, 'transacciones.parquet'))
    df_cli = pd.read_parquet(os.path.join(raw_dir, 'clientes.parquet'))
    df_prod = pd.read_parquet(os.path.join(raw_dir, 'productos.parquet'))

    # 2) Eliminar duplicados exactos y trabajar con copias seguras
    df_trans_clean = df_trans.copy().drop_duplicates()
    df_clientes_clean = df_cli.copy().drop_duplicates()
    df_prod_clean = df_prod.copy().drop_duplicates()

    # 3) Normalizar tipos de ID a string para evitar conflictos en el merge
    df_trans_clean["customer_id"] = df_trans_clean["customer_id"].astype(str)
    df_trans_clean["product_id"] = df_trans_clean["product_id"].astype(str)
    df_clientes_clean["customer_id"] = df_clientes_clean["customer_id"].astype(str)
    df_prod_clean["product_id"] = df_prod_clean["product_id"].astype(str)

    # 4) Convertir la fecha de compra a datetime
    df_trans_clean["purchase_date"] = pd.to_datetime(df_trans_clean["purchase_date"])

    # 5) Merge: primero transacciones + clientes, luego resultado + productos
    df_merged = (
        df_trans_clean
        .merge(df_clientes_clean, on="customer_id", how="left")
        .merge(df_prod_clean, on="product_id", how="left")
    )

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'merged.parquet')
    df_merged.to_parquet(out_path, index=False)
    
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
    df_merged['year_week'] = df_merged['purchase_date'].dt.to_period('W').astype(str)

    # Sumar ítems por semana
    df_semanal = (
        df_merged
        .groupby(['customer_id', 'product_id', 'year_week'], as_index=False)
        .agg(items_sum=('items', 'sum'))
    )

    # Obtener pares únicos (customer_id, product_id)
    pares_unicos = df_merged[['customer_id', 'product_id']].drop_duplicates()

    # Obtener todas las semanas completas en el rango de fechas
    fechas_completas = pd.date_range(df_merged['purchase_date'].min(), df_merged['purchase_date'].max(), freq='D')
    semanas_completas = pd.Series(fechas_completas.to_period('W').unique().astype(str), name='year_week')
    
    # Mapeo de 'YYYYWXX' a 'week_N'
    semanas_completas_map = {week : f'week_{i+1}' for i, week in enumerate(semanas_completas)}

    # Crear todas las combinaciones válidas de pares y semanas
    # Este enfoque es más robusto que MultiIndex.from_product directo si hay IDs con diferente rango de fechas
    combinaciones_validas = pd.MultiIndex.from_frame(
        pares_unicos.assign(key=1).merge(semanas_completas.to_frame().assign(key=1), on='key').drop('key', axis=1)
    )

    # Reindexar para asegurar todas las combinaciones posibles y rellenar con 0
    df_semanal_idx = df_semanal.set_index(['customer_id', 'product_id', 'year_week'])
    df_full = df_semanal_idx.reindex(combinaciones_validas, fill_value=0).reset_index()
    
    # Generar la etiqueta (1 si items_sum > 0, 0 en caso contrario)
    df_full['label'] = (df_full['items_sum'] > 0).astype(int)
    
    # Renombrar 'year_week' a 'week_N'
    df_full['year_week'] = df_full['year_week'].map(semanas_completas_map)
    
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'labeled.parquet')
    df_full.to_parquet(out_path, index=False)
    
    return out_path


def create_label_new_data(merge_path: str, i: int) -> str:
    """
    Genera etiquetas (0/1) para un nuevo conjunto de datos (una sola semana),
    asignando un número de semana basado en el índice 'i' y guarda labeled_week_{i}.parquet.

    Args:
        merge_path (str): Ruta al archivo Parquet combinado de entrada para la nueva semana.
        i (int): El número de semana a asignar a todos los registros en este archivo.

    Returns:
        str: La ruta donde se guardó el archivo Parquet etiquetado.
    """
    _, processed_dir,_ = get_dirs()
    df = pd.read_parquet(merge_path)

    # Normalizar tipos de ID a string, aunque ya deberían venir limpios de extract_and_merge
    df["customer_id"] = df["customer_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    
    # Asignar directamente el nombre de la semana basado en 'i'
    df['year_week'] = f'week_{i}'

    # Sumar ítems por cliente, producto y la semana asignada
    df_semanal = df.groupby(
        ['customer_id', 'product_id', 'year_week'], as_index=False
    ).agg(items_sum=('items', 'sum'))

    # Combinaciones completas de customer_id y product_id existentes en este DataFrame
    pares = df[['customer_id', 'product_id']].drop_duplicates()
    
    # Crear un MultiIndex con todas las combinaciones de pares y la única semana 'i'
    idx = pd.MultiIndex.from_product(
        [pares['customer_id'].unique(), pares['product_id'].unique(), [f'week_{i}']],
        names=['customer_id', 'product_id', 'year_week']
    )

    # Reindexar para asegurar todas las combinaciones posibles para esta semana 'i'
    df_full = df_semanal.set_index(['customer_id', 'product_id', 'year_week'])\
                        .reindex(idx, fill_value=0).reset_index()

    # Generar la etiqueta (1 si items_sum > 0, 0 en caso contrario)
    df_full['label'] = (df_full['items_sum'] > 0).astype(int)

    os.makedirs(processed_dir, exist_ok=True)
    out_filename = f'labeled_week_{i}.parquet'
    out_path = os.path.join(processed_dir, out_filename)
    df_full.to_parquet(out_path, index=False)
    
    return out_path


def prepare_model_dataset(merge_path: str, labeled_path: str) -> list[str]:
    """
    Construye dataset con atributos constantes y filtra tras primera compra.
    Guarda datasets separados por semana en './data/processed/week_data/'.

    Args:
        merge_path (str): Ruta al archivo Parquet combinado de entrada (raw data).
        labeled_path (str): Ruta al archivo Parquet etiquetado (puede ser 'labeled.parquet'
                            o 'labeled_week_X.parquet' para datos nuevos).

    Returns:
        list[str]: Una lista de rutas a los archivos Parquet semanales guardados.
    """
    _, processed_dir,_ = get_dirs()
    df_merged = pd.read_parquet(merge_path)
    df_label  = pd.read_parquet(labeled_path)

    # 1) Extraer atributos constantes de cliente y producto
    df_client = (
        df_merged[['customer_id','region_id','zone_id','customer_type',
                   'Y','X','num_deliver_per_week','num_visit_per_week']]
        .drop_duplicates(subset='customer_id') # Usar subset para mayor claridad
    )
    df_prod = (
        df_merged[['product_id','brand','category','sub_category','segment','package','size']]
        .drop_duplicates(subset='product_id') # Usar subset para mayor claridad
    )

    # 2) Unir df_label con atributos de cliente y producto
    df_model = (
        df_label
        .merge(df_client, on='customer_id', how='left')
        .merge(df_prod, on='product_id', how='left')
    )

    # 3) Eliminar cualquier columna de fecha que pueda quedar del merge
    df_model = df_model.drop(columns=[c for c in df_model.columns if 'purchase_date' in c], errors='ignore')

    # 4) Filtrar tras primera compra
    # Asegurarse de que 'year_week' sea un entero para la comparación
    df_model['year_week'] = df_model['year_week'].astype(str).str.replace('week_','').astype(int)
    
    def filter_fun(group):
        purchase = group[group['label'] == 1]
        if purchase.empty:
            return pd.DataFrame() # Si no hay compras, no hay registros para mantener

        # Obtener la semana de la primera compra para este par (customer_id, product_id)
        primera_semana = purchase['year_week'].min()
        
        # Mantener solo los registros a partir de la primera compra (o si hay una compra)
        return group[(group['label'] == 1) | (group['year_week'] >= primera_semana)]

    df_modelo_filtrado = df_model.groupby(['customer_id', 'product_id'], group_keys=False).apply(filter_fun)

    # Añadir 'week_num' para compatibilidad con la lógica de batches (aunque no se use directamente aquí)
    # y para la función de guardar.
    df_modelo_filtrado['week_num'] = df_modelo_filtrado['year_week'].astype(int)

    df_modelo_filtrado.to_parquet(processed_dir+'/filtado.parquet')
    # 5) Llamar a save_weekly_datasets para guardar por semana
    saved_paths = save_weekly_datasets(df_modelo_filtrado)
    
    return saved_paths


def save_weekly_datasets(df_filtered: pd.DataFrame) -> list[str]:
    """
    Guarda el dataset filtrado como archivos Parquet separados por semana
    en el directorio './data/processed/week_data/'.

    Args:
        df_filtered (pd.DataFrame): El DataFrame preparado por prepare_model_dataset,
                                    que debe contener la columna 'year_week' (como entero).

    Returns:
        list[str]: Una lista de rutas a los archivos Parquet semanales guardados.
    """
    _, processed_dir, week_data_dir = get_dirs()
    #week_data_dir = os.path.join(processed_dir, 'week_data')
    os.makedirs(week_data_dir, exist_ok=True)
    saved_paths = []

    # Asegurarse de que 'year_week' sea un string 'week_X' para el nombre del archivo
    # Usamos la columna 'week_num' que ya es int, para generar el nombre.
    df_filtered['year_week_str'] = 'week_' + df_filtered['week_num'].astype(str)

    # Agrupar por semana y guardar cada grupo como un archivo parquet separado
    for week_name, week_df in df_filtered.groupby('year_week_str'):
        # Eliminar columnas temporales o no necesarias antes de guardar
        week_df_to_save = week_df.drop(columns=['year_week_str', 'week_num'], errors='ignore')
        
        out_path = os.path.join(week_data_dir, f'{week_name}.parquet')
        week_df_to_save.to_parquet(out_path, index=False)
        saved_paths.append(out_path)
        # No hay prints aquí, según la solicitud.

    return saved_paths


# ---------- Transformaciones y estandarización ----------
drop_zero_var = ['region_id','zone_id','num_visit_per_week','items_sum']
def filter_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra rango geográfico X/Y"""
    return df[(df['X']>-108)&(df['X']<-107)&(df['Y']>-48)&(df['Y']<-46)]

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra, elimina variables constantes, extrae semana,
    one-hot semanas, escala numérico y codifica categóricas.
    """
    df2 = filter_xy(df.copy()).drop(columns=drop_zero_var, errors='ignore')
    # Manejo de 'year_week'
    if pd.api.types.is_integer_dtype(df2['year_week']):
        df2['week_num'] = df2['year_week']
    else:
        df2['week_num'] = df2['year_week'].astype(str).str.extract(r'week_(\\d+)')[0].astype(int)
    # One-hot de semana
    week_ohe = pd.get_dummies(df2['week_num'], prefix='week')
    # Numéricas
    num_cols = ['size','X','Y']
    df_num = df2[num_cols].fillna(df2[num_cols].median())
    means, stds = df_num.mean(), df_num.std()
    df_num_scaled = (df_num - means) / stds
    df_num_scaled.columns = num_cols
    # One-hot de entregas
    deliver_ohe = pd.get_dummies(df2['num_deliver_per_week'], prefix='deliver')
    # Categóricas
    cat_cols = ['customer_type','brand','category','sub_category','segment','package']
    df_cat = df2[cat_cols].fillna('missing')
    cat_ohe = pd.get_dummies(df_cat, prefix=cat_cols)
    # Combina features
    result = pd.concat([week_ohe, df_num_scaled, deliver_ohe, cat_ohe], axis=1)
    result['week_num'] = df2['week_num']
    return result


def split_train_val_windows(dataset_path: str) -> list[str]:
    """
    Divide el dataset de features en archivos semanales por week_num.
    Guarda cada subset en data/week_data/week_{i}.parquet.

    Args:
        dataset_path (str): Ruta al Parquet con el dataset de features.
    Returns:
        list[str]: Lista de rutas a los archivos Parquet generados.
    """
    _, _, week_data_dir = get_dirs()
    os.makedirs(week_data_dir, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    if 'week_num' not in df.columns:
        raise KeyError("La columna 'week_num' no existe en el dataset de features")

    paths = []
    for week in sorted(df['week_num'].unique()):
        df_week = df[df['week_num'] == week]
        out_file = os.path.join(week_data_dir, f'week_{week}.parquet')
        df_week.to_parquet(out_file, index=False)
        paths.append(out_file)

    return paths


def save_weekly_features_labels(filtered_path: str) -> dict[str, list[str]]:
    """
    Crea, bajo data/processed, dos carpetas: 'features' y 'labels'.
    - En 'features/' guarda por semana week_{i}.parquet con todas las columnas
      del df_filtrado excepto 'label'.
    - En 'labels/' guarda label_week_{i}.parquet con solo la columna 'label'
      del mismo subset y en el mismo orden.

    Args:
        filtered_path: ruta al Parquet generado por prepare_model_dataset.
    Returns:
        dict: {
          'features': [ruta_a_week_1.parquet, …, ruta_a_week_N.parquet],
          'labels':   [ruta_a_label_week_1.parquet, …, ruta_a_label_week_N.parquet]
        }
    """
    import os
    import pandas as pd

    _, processed_dir, _ = get_dirs()
    features_dir = os.path.join(processed_dir, 'features')
    labels_dir   = os.path.join(processed_dir, 'labels')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir,   exist_ok=True)

    df = pd.read_parquet(filtered_path)
    # Asegúrate de tener week_num
    if 'week_num' not in df.columns:
        if pd.api.types.is_string_dtype(df['year_week']):
            df['week_num'] = df['year_week'].str.extract(r'week_(\d+)')[0].astype(int)
        else:
            df['week_num'] = df['year_week'].astype(int)

    feat_paths = []
    label_paths = []

    for week in sorted(df['week_num'].unique()):
        df_week = df[df['week_num'] == week]
        # Features (todo excepto label)
        feat_df = df_week.drop(columns=['label'], errors='ignore')
        feat_file = os.path.join(features_dir, f'week_{week}.parquet')
        feat_df.to_parquet(feat_file, index=False)
        feat_paths.append(feat_file)

        # Labels (solo la columna label)
        lbl_df = df_week[['label']]
        lbl_file = os.path.join(labels_dir, f'label_week_{week}.parquet')
        lbl_df.to_parquet(lbl_file, index=False)
        label_paths.append(lbl_file)

    return {'features': feat_paths, 'labels': label_paths}


if __name__ == '__main__':
    mp = extract_and_merge()
    print(f"Merged at: {mp}")
    lp = create_label(mp)
    print(f"Labeled at: {lp}")
    final = prepare_model_dataset(mp, lp)
    print(f"Filtered dataset at: {final}")