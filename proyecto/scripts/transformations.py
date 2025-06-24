import os
import pandas as pd

def get_dirs():
    """
    Determina rutas raw y processed según AIRFLOW_HOME.
    """
    airflow_home = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
    raw_dir = os.path.join(airflow_home, 'data', 'raw')
    processed_dir = os.path.join(airflow_home, 'data', 'processed')
    return raw_dir, processed_dir


def extract_and_merge() -> str:
    """
    Lee Parquet de transacciones, clientes y productos, los fusiona
    y guarda merged.parquet en processed.
    """
    raw_dir, processed_dir = get_dirs()
    df_trans = pd.read_parquet(os.path.join(raw_dir, 'transacciones.parquet'))
    df_cli   = pd.read_parquet(os.path.join(raw_dir, 'clientes.parquet'))
    df_prod  = pd.read_parquet(os.path.join(raw_dir, 'productos.parquet'))
    df_merged = (
        df_trans
        .merge(df_cli, on='customer_id', how='left')
        .merge(df_prod, on='product_id', how='left')
    )
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'merged.parquet')
    df_merged.to_parquet(out_path, index=False)
    return out_path


def create_label(merge_path: str) -> str:
    """
    Genera etiquetas semanales (0/1) y guarda labeled.parquet.
    """
    _, processed_dir = get_dirs()
    df = pd.read_parquet(merge_path)
    # año-semana
    df['year_week'] = df['purchase_date'].dt.to_period('W').astype(str)
    # suma items por semana
    df_semanal = df.groupby(
        ['customer_id','product_id','year_week'], as_index=False
    ).agg(items_sum=('items','sum'))
    # combinaciones completas
    pares = df[['customer_id','product_id']].drop_duplicates()
    semanas = pd.Series(
        pd.date_range(df['purchase_date'].min(), df['purchase_date'].max(), freq='D')
        .to_period('W').unique().astype(str), name='year_week'
    )
    idx = pd.MultiIndex.from_product(
        [pares['customer_id'].unique(), pares['product_id'].unique(), semanas],
        names=['customer_id','product_id','year_week']
    )
    df_full = df_semanal.set_index(['customer_id','product_id','year_week']).reindex(idx, fill_value=0).reset_index()
    df_full['label'] = (df_full['items_sum']>0).astype(int)
    # renombrar semana
    week_map = {w: f'week_{i+1}' for i,w in enumerate(semanas)}
    df_full['year_week'] = df_full['year_week'].map(week_map)
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'labeled.parquet')
    df_full.to_parquet(out_path, index=False)
    return out_path


def prepare_model_dataset(merge_path: str, labeled_path: str) -> str:
    """
    Construye dataset con atributos constantes y filtra tras primera compra.
    Guarda dataset_filtrado.parquet.
    """
    _, processed_dir = get_dirs()
    df_merged = pd.read_parquet(merge_path)
    df_label  = pd.read_parquet(labeled_path)
    # atributos constantes
    df_client = df_merged[['customer_id','region_id','zone_id','customer_type',
                           'Y','X','num_deliver_per_week','num_visit_per_week']]
    df_client = df_client.drop_duplicates('customer_id')
    df_prod = df_merged[['product_id','brand','category','sub_category','segment','package','size']]
    df_prod = df_prod.drop_duplicates('product_id')
    # unir
    df_model = df_label.merge(df_client, on='customer_id', how='left')
    df_model = df_model.merge(df_prod, on='product_id', how='left')
    # eliminar cols fecha
    df_model = df_model.drop(columns=[c for c in df_model.columns if 'purchase_date' in c], errors='ignore')
    # filtrar tras primera compra
    df_model['year_week'] = df_model['year_week'].astype(str).str.replace('week_','').astype(int)
    def filter_fun(group):
        compra = group[group['label']==1]
        if compra.empty:
            return pd.DataFrame()
        first = compra['year_week'].min()
        return group[(group['label']==1)|(group['year_week']>=first)]
    df_filtrado = df_model.groupby(['customer_id','product_id'], group_keys=False).apply(filter_fun)
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'dataset_filtrado.parquet')
    df_filtrado.to_parquet(out_path, index=False)
    return out_path


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
    return result


def split_train_val_windows(dataset_path: str,
                            time_window: int = 4,
                            stride: int = 3,
                            week_col: str = 'year_week') -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Genera ventanas de train/val con tamaño time_window y desplazamiento stride.
    Args:
        dataset_path: ruta al Parquet con dataset_filtrado.
        time_window: número de semanas por ventana.
        stride: paso entre ventanas.
        week_col: columna que contiene año_semana (string "week_i" o entero i).

    Returns:
        splits: lista de tuplas (train_df, val_df) por cada ventana.
    """
    _, _ = get_dirs()
    df = pd.read_parquet(dataset_path)

    # 1) Detectar si year_week es string o entero y extraer week_num:
    col = df[week_col]
    if pd.api.types.is_string_dtype(col):
        # cadena “week_1”, “week_2”, …
        week_nums = col.str.extract(r'week_(\d+)')[0].astype(int)
    else:
        # ya es entero (p.ej. 1, 2, 3, …)
        week_nums = col.astype(int)
    df['week_num'] = week_nums

    # 2) Crear ventanas deslizantes
    weeks = sorted(df['week_num'].unique())
    batches = [weeks[i : i + time_window]
               for i in range(0, len(weeks) - time_window + 1, stride)]

    # 3) Manejo de remanente final
    rem = weeks[len(weeks) - time_window + stride :]
    if rem:
        if len(rem) <= time_window // 2:
            batches[-1].extend(rem)
        else:
            batches.append(rem)

    # 4) Generar lista de splits (train, val)
    splits = []
    split_pt = time_window - stride
    for batch in batches:
        train_weeks = batch[:split_pt]
        val_weeks   = batch[split_pt:]
        train_df = df[df['week_num'].isin(train_weeks)]
        val_df   = df[df['week_num'].isin(val_weeks)]
        splits.append((train_df, val_df))

    return splits





# Batches (opcional reutilización)
class Batch:
    """
    Divide datos en ventanas con stride para train/val.
    """
    def __init__(self, data: pd.DataFrame, time_window_in_weeks: int, stride: int, col: str='week_num'):
        self.data = data.copy()
        self.time_window = time_window_in_weeks
        self.stride = stride
        self.col = col
        self._prepare()
    def _prepare(self):
        weeks = sorted(self.data[self.col].unique())
        n = len(weeks)
        self.weeks_batches = [weeks[i:i+self.time_window] for i in range(0, n-self.time_window+1, self.stride)]
        rem = weeks[n-self.time_window+self.stride:]
        if rem:
            if len(rem)<=self.time_window//2:
                self.weeks_batches[-1].extend(rem)
            else:
                self.weeks_batches.append(rem)
    def get_train_eval_batch(self, i:int):
        batch = self.weeks_batches[i]
        split = self.time_window-self.stride
        train = self.data[self.data[self.col].isin(batch[:split])]
        val   = self.data[self.data[self.col].isin(batch[split:])]
        return train, val
    def get_test_batch(self, week:int): return self.data[self.data[self.col]==week]
    def __len__(self): return len(self.weeks_batches)
    def __getitem__(self,index:int): return self.get_train_eval_batch(index)
    def __iter__(self):
        for i in range(len(self)): yield self.get_train_eval_batch(i)


if __name__ == '__main__':
    mp = extract_and_merge()
    print(f"Merged at: {mp}")
    lp = create_label(mp)
    print(f"Labeled at: {lp}")
    final = prepare_model_dataset(mp, lp)
    print(f"Filtered dataset at: {final}")