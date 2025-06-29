from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, FunctionTransformer, KBinsDiscretizer,
)

class WeekExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nada que ajustar

    def transform(self, X):
        week_nums = X['year_week'].str.extract(r'week_(\d+)')[0].astype(int)
        return pd.DataFrame({'week_num': week_nums}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(['week_num'])

class MedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X.median()
        return self

    def transform(self, X):
        return X.fillna(self.medians_)

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else X.columns.to_list()
  

class ColumnReshaper(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.column]].values

    def get_feature_names_out(self, input_features=None):
        return [self.column]
    

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value='missing'):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(self.fill_value)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        else:
            return []
        

# ---------- A) Transformaciones ----------
drop_zero_var = ['region_id', 'zone_id', 'num_visit_per_week', 'items_sum']

def filter_xy(df):
    return df[
        (df['X'] > -108) & (df['X'] < -107) &
        (df['Y'] > -48)  & (df['Y'] < -46)
    ]

week_transformer = Pipeline([
    ('extract_week', WeekExtractor()),
    ('ohe_week', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numeric_transformer = Pipeline([
    ('impute_num', MedianImputer()),
    ('scale_num',  StandardScaler())
])

deliver_binner = Pipeline([
    ('reshape', ColumnReshaper('num_deliver_per_week')),
    ('kbins',   OneHotEncoder(handle_unknown='ignore', sparse_output=False,))
])

categorical_cols = ['customer_type','brand','category','sub_category','segment','package']
categorical_transformer = Pipeline([
    ('impute_cat', CategoricalImputer()),
    ('ohe',        OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    #('week_ohe',     week_transformer,        ['year_week']),  # NO LO USAMOS PORQUE NO SABEMOS EL LARGO DE LAS SEMANAS A PRIORI (pueden ser 100 o 53).
    ('num',          numeric_transformer,     ['size','X','Y']),
    ('deliver_bins', deliver_binner,          ['num_deliver_per_week']),
    ('cat',          categorical_transformer, categorical_cols),
    ('week_passthrough', 'passthrough', ['week_num', 'label'])
], remainder='drop', verbose_feature_names_out=False)

pipeline_fe = Pipeline([
    ('preprocessor', preprocessor)
])
