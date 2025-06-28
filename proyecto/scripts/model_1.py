import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, f1_score
import logging
import coloredlogs
from collections import deque
import os
import joblib 
from datetime import datetime, timedelta

# Configuración básica de logs
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger) 

# --- Rutas de Archivos ---
MODEL_STATE_PATH = './data/processed/model_state/incremental_xgboost_model.pkl'
FEATURES_DIR = './data/processed/features/'
LABELS_DIR = './data/processed/labels/'
CURRENT_PREDICTION_WEEK_FILE = './data/processed/model_state/current_prediction_week.txt'
PREDICTIONS_DIR = './data/processed/predictions/' 

class IncrementalXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, f1_threshold_drop: float = 0.05,
                 reset_window_size: int = 8, 
                 initial_training_weeks: int = 4, 
                 max_depth: int = 6, n_estimators: int = 100, learning_rate: float = 0.1,
                 early_stopping_rounds: int = 12,
                 scale_pos_weight: float = 1.0): 
        
        self.f1_threshold_drop = f1_threshold_drop
        self.reset_window_size = reset_window_size
        self.initial_training_weeks = initial_training_weeks
        
        self.xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'tree_method': 'auto',
            'early_stopping_rounds': early_stopping_rounds,
            'scale_pos_weight': scale_pos_weight,
            'n_jobs': -1 
        }
        
        self.model: XGBClassifier = None 
        self.history_: deque = deque(maxlen=reset_window_size) 
        self.last_evaluated_f1: float = None 
        self.label_col: str = 'label_col' 

    def _log(self, level, message):
        logger.log(level, f"[{self.__class__.__name__}] {message}")

    def _create_or_update_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_eval: pd.DataFrame = None, y_eval: pd.Series = None, 
                                is_full_retrain: bool = False) -> XGBClassifier:
        
        self._log(logging.INFO, f"Initiating XGBoost training/update. Train shape: {X_train.shape}, Eval shape: {X_eval.shape}")
        
        fit_params = {
            #'eval_set': [(X_eval, y_eval)],
            'callbacks': [self._early_stopping_callback]
        }

        if is_full_retrain or self.model is None:
            self.model = XGBClassifier(**self.xgb_params)
            self._log(logging.INFO, "New XGBoost model created for full retraining or cold start.")
            self.model.fit(X_train, y_train, **fit_params)
        else:
            self._log(logging.INFO, "Incrementally updating existing XGBoost model.")
            self.model.fit(X_train, y_train, xgb_model=self.model.get_booster(), **fit_params) 

        self._log(logging.INFO, "XGBoost training/update completed.")
        return self.model


    def _early_stopping_callback(self, env):
        pass 


    def _evaluate_performance(self, model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
        if X.empty or y.empty:
            self._log(logging.WARNING, "No data for performance evaluation. Returning 0.0 for AUC and F1-score.")
            return 0.0, 0.0
        
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc_score = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)
        
        self._log(logging.INFO, f"Evaluation results: AUC={auc_score:.4f}, F1-score={f1:.4f}")
        return auc_score, f1


    def initial_train(self):
        """
        Performs initial training (cold start) of the model using historical data.
        This run does not perform an evaluation based on the DAG's new invariant.
        """
        self._log(logging.INFO, "Starting cold start training.")
        
        all_feature_files = sorted([f for f in os.listdir(FEATURES_DIR) if f.startswith('features_week_') and f.endswith('.parquet')])
        all_label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.startswith('labels_week_') and f.endswith('.parquet')])

        if len(all_feature_files) < self.initial_training_weeks or \
           len(all_label_files) < self.initial_training_weeks:
            self._log(logging.ERROR, f"Insufficient historical data for cold start. Found {len(all_feature_files)} weeks, required {self.initial_training_weeks}.")
            raise ValueError(f"Insufficient historical data for cold start. Required {self.initial_training_weeks} weeks.")
        
        initial_dfs = []
        for i in range(self.initial_training_weeks):
            feature_path = os.path.join(FEATURES_DIR, all_feature_files[i])
            label_path = os.path.join(LABELS_DIR, all_label_files[i])
            
            features = pd.read_parquet(feature_path)
            labels = pd.read_parquet(label_path)
            
            merged_df = pd.merge(features, labels, on=['customer_id', 'product_id', 'week_num'], how='left') 
            
            if self.xgb_params['objective'] == 'binary:logistic':
                merged_df[self.label_col] = merged_df[self.label_col].fillna(0).astype(int) 
            
            initial_dfs.append(merged_df)
            self.history_.append(merged_df) 
        
        # Para el entrenamiento inicial (cold start), usamos todas las semanas disponibles en initial_dfs para entrenar
        combined_df = pd.concat(initial_dfs, ignore_index=True)
        X_train_initial = combined_df.drop(columns=[self.label_col, 'customer_id', 'product_id'], errors='ignore')
        y_train_initial = combined_df[self.label_col]
        
        # Le pasamos como evalset los datos de entrenamiento. Esto para early stopping.
        X_eval_initial = X_train_initial.copy() 
        y_eval_initial = y_train_initial.copy()

        self._create_or_update_model(X_train_initial, y_train_initial, X_eval_initial, y_eval_initial, is_full_retrain=True)
        self._log(logging.INFO, "Cold start training completed. Model is now ready for predictions.")

        last_trained_week_str = all_label_files[self.initial_training_weeks - 1].replace('labels_week_', '').replace('.parquet', '')
        last_trained_date = datetime.strptime(last_trained_week_str, '%Y%m%d')
        next_prediction_date = last_trained_date + timedelta(weeks=1)
        next_prediction_week_str = next_prediction_date.strftime('%Y%m%d')
        IncrementalXGBoost.set_current_week_to_predict(next_prediction_week_str)
        self._log(logging.INFO, f"Next week for prediction initialized to: {next_prediction_week_str}.")


    # --- MÉTODOS PARA EL FLUJO DE TRABAJO DEL DAG ---

    def train_incremental_and_update_history(self, week_t_date_str: str):
        """
        Trains or incrementally updates the model with data from week T.
        This method is designed to be run when labels for week T become available.
        Args:
            week_t_date_str (str): Date of Week T in 'YYYYMMDD' format. This is the latest week for which labels are available.
        """
        self._log(logging.INFO, f"Starting incremental training for Week {week_t_date_str}.")

        features_path = os.path.join(FEATURES_DIR, f"features_week_{week_t_date_str}.parquet")
        labels_path = os.path.join(LABELS_DIR, f"labels_week_{week_t_date_str}.parquet")

        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            self._log(logging.ERROR, f"Missing features or labels for Week {week_t_date_str} at {features_path} or {labels_path}.")
            raise FileNotFoundError(f"Data for Week {week_t_date_str} not found for training.")

        features_t = pd.read_parquet(features_path)
        labels_t = pd.read_parquet(labels_path)
        
        full_week_t_df = pd.merge(features_t, labels_t, on=['customer_id', 'product_id', 'week_num'], how='left') 
        
        if self.xgb_params['objective'] == 'binary:logistic':
            full_week_t_df[self.label_col] = full_week_t_df[self.label_col].fillna(0).astype(int) 

        # Añadimos la semana actual al historial. Esto también gestiona la ventana deslizante.
        self.history_.append(full_week_t_df)

        # Usamos la semana actual (T) para el entrenamiento incremental.
        X_train = full_week_t_df.drop(columns=[self.label_col, 'customer_id', 'product_id'], errors='ignore')
        y_train = full_week_t_df[self.label_col]
        
        # Para el entrenamiento incremental, usamos los mismos datos para el eval_set para el early stopping.
        X_eval = X_train.copy() 
        y_eval = y_train.copy()

        self._create_or_update_model(X_train, y_train, X_eval, y_eval, is_full_retrain=False)
        self._log(logging.INFO, f"Incremental training for Week {week_t_date_str} completed. Model updated.")


    def generate_predictions_for_next_week(self) -> pd.DataFrame:
        """
        Generates probability predictions for the 'current prediction week' (N+1).
        It takes the features from the 'last trained week' (N) and updates their 'week_num' to N+1.
        The current prediction week is managed by the static counter.
        Returns:
            pd.DataFrame: DataFrame with 'customer_id', 'product_id', 'week_num', 'prediction_proba'.
        """
        prediction_week_str = IncrementalXGBoost.get_current_week_to_predict()
        if not prediction_week_str:
            self._log(logging.ERROR, "No 'current prediction week' found. Cannot generate predictions.")
            raise RuntimeError("Current prediction week is not set. Run initial_train first or set it manually.")

        self._log(logging.INFO, f"Generating predictions for Week {prediction_week_str}.")

        if self.model is None:
            self._log(logging.ERROR, "Model not trained. Cannot make predictions.")
            raise RuntimeError("Model is not trained. Please train it first.")
        
        # La "última semana entrenada" es la última en el historial.
        # Si el historial está vacío, significa que el modelo no ha sido entrenado.
        if not self.history_:
            self._log(logging.ERROR, "Model history is empty. Cannot determine 'last trained week' for prediction features.")
            raise RuntimeError("Model history is empty. Ensure training has occurred.")
        
        # Tomar los features de la última semana EN EL HISTORIAL (que es la última entrenada)
        last_trained_week_df = self.history_[-1].copy() # Usamos una copia para no modificar el df original en el historial
        
        # Actualizar 'week_num' a la semana de predicción (N+1)
        if 'week_num' in last_trained_week_df.columns:
            last_trained_week_df['week_num'] = int(prediction_week_str) # Aseguramos que sea entero si así lo espera el modelo
        else:
            self._log(logging.WARNING, "'week_num' column not found in last trained data. Predictions will proceed without this feature.")

        ids = last_trained_week_df[['customer_id', 'product_id', 'week_num']].copy() 
        X_predict_model = last_trained_week_df.drop(columns=[self.label_col, 'customer_id', 'product_id'], errors='ignore')
        
        probabilities = self.model.predict_proba(X_predict_model)[:, 1]
        
        predictions_df = ids
        predictions_df['prediction_proba'] = probabilities
        
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        prediction_output_path = os.path.join(PREDICTIONS_DIR, f"predictions_week_{prediction_week_str}.parquet")
        predictions_df.to_parquet(prediction_output_path, index=False)
        self._log(logging.INFO, f"Predictions for Week {prediction_week_str} saved to {prediction_output_path}.")

        self._log(logging.INFO, f"Predictions generated for {len(predictions_df)} customer-product pairs for Week {prediction_week_str}.")
        return predictions_df
    
    def evaluate_and_detect_drift(self):
        """
        Evaluates predictions for the 'current prediction week' (N+1) once actual labels are available.
        This method is designed to be run AFTER generate_predictions_for_next_week and when labels are ready.
        It also handles data drift detection and updates the static prediction week counter.
        Returns:
            float: F1-score of the evaluated predictions.
        """
        week_to_evaluate_str = IncrementalXGBoost.get_current_week_to_predict()
        if not week_to_evaluate_str:
            self._log(logging.ERROR, "No 'current prediction week' found for evaluation.")
            raise RuntimeError("Current prediction week is not set. Cannot evaluate.")

        self._log(logging.INFO, f"Starting evaluation and drift detection for Week {week_to_evaluate_str}.")

        predictions_path = os.path.join(PREDICTIONS_DIR, f"predictions_week_{week_to_evaluate_str}.parquet")
        labels_path = os.path.join(LABELS_DIR, f"labels_week_{week_to_evaluate_str}.parquet")

        if not os.path.exists(predictions_path) or not os.path.exists(labels_path):
            self._log(logging.INFO, f"Data (predictions or labels) for Week {week_to_evaluate_str} not yet fully available.")
            # Este es el punto de pausa en el DAG. El DAG debería tener una condición aquí.
            return None # O lanzar una excepción específica para el orquestador

        predictions_df = pd.read_parquet(predictions_path)
        actual_labels_df = pd.read_parquet(labels_path)

        if self.xgb_params['objective'] == 'binary:logistic':
            actual_labels_df[self.label_col] = actual_labels_df[self.label_col].fillna(0).astype(int)

        eval_df = pd.merge(predictions_df, actual_labels_df, on=['customer_id', 'product_id', 'week_num'], how='left')
        eval_df.dropna(subset=[self.label_col, 'prediction_proba'], inplace=True)

        if eval_df.empty:
            self._log(logging.WARNING, f"No valid data points for evaluation for Week {week_to_evaluate_str} after merging predictions and labels. F1-score set to 0.0.")
            self.last_evaluated_f1 = 0.0
        else:
            y_true = eval_df[self.label_col]
            y_pred_proba = eval_df['prediction_proba']
            f1 = f1_score(y_true, (y_pred_proba > 0.5).astype(int))
            auc_score = roc_auc_score(y_true, y_pred_proba)
            self._log(logging.INFO, f"Evaluation for Week {week_to_evaluate_str} completed. AUC={auc_score:.4f}, F1-score={f1:.4f}")
            self.last_evaluated_f1 = f1

        # --- Lógica de Detección de Data Drift ---
        
        # --- Actualizar el contador estático para la próxima semana de predicción ---
        current_evaluated_date = datetime.strptime(week_to_evaluate_str, '%Y%m%d')
        next_prediction_date = current_evaluated_date + timedelta(weeks=1)
        next_prediction_week_str = next_prediction_date.strftime('%Y%m%d')
        IncrementalXGBoost.set_current_week_to_predict(next_prediction_week_str)
        self._log(logging.INFO, f"Static prediction week counter updated to {next_prediction_week_str}.")

        return self.last_evaluated_f1 
    
    def full_retrain(self):
        """
        Performs a full retraining of the model using the entire historical window.
        Triggered when data drift is detected or as a scheduled full refresh.
        """
        self._log(logging.INFO, "Starting full model retraining due to detected data drift or scheduled refresh.")
        
        if not self.history_:
            self._log(logging.ERROR, "Cannot perform full retraining: model history is empty.")
            raise ValueError("Model history is empty for full retraining.")

        combined_history_df = pd.concat(list(self.history_), ignore_index=True)
        
        # Usa todo el historial para reentrenar completamente
        X_train_full = combined_history_df.drop(columns=[self.label_col, 'customer_id', 'product_id'], errors='ignore')
        y_train_full = combined_history_df[self.label_col]
        
        # Para la evaluación interna durante el fit del full_retrain, usamos el mismo set.
        X_eval_full = X_train_full.copy()
        y_eval_full = y_train_full.copy()

        self._create_or_update_model(X_train_full, y_train_full, X_eval_full, y_eval_full, is_full_retrain=True)
        
        _, self.last_evaluated_f1 = self._evaluate_performance(self.model, X_eval_full, y_eval_full)
        self._log(logging.INFO, f"Full retraining completed. New F1-score after retraining: {self.last_evaluated_f1:.4f}")

        return self.last_evaluated_f1 
    
    @staticmethod
    def load_model(path=MODEL_STATE_PATH):
        """Loads the IncrementalXGBoost instance from a file."""
        if os.path.exists(path):
            try:
                model_instance = joblib.load(path)
                logger.info(f"IncrementalXGBoost model loaded from {path}.")
                return model_instance
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}. Initializing a new model instance.")
                return IncrementalXGBoost() 
        else:
            logger.info(f"No existing model found at {path}. Creating a new IncrementalXGBoost instance.")
            return IncrementalXGBoost() 

    def save_model(self, path=MODEL_STATE_PATH):
        """Saves the current IncrementalXGBoost instance to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        joblib.dump(self, path)
        self._log(logging.INFO, f"IncrementalXGBoost model saved to {path}.")

    @staticmethod
    def set_current_week_to_predict(week_str: str):
        """
        Sets the current week that needs to be predicted.
        This is stored in a simple text file.
        Args:
            week_str (str): The week in 'YYYYMMDD' format.
        """
        try:
            os.makedirs(os.path.dirname(CURRENT_PREDICTION_WEEK_FILE), exist_ok=True)
            with open(CURRENT_PREDICTION_WEEK_FILE, 'w') as f:
                f.write(week_str.strip())
            logger.info(f"Static prediction week counter set to: {week_str}.")
        except Exception as e:
            logger.error(f"Error saving current prediction week '{week_str}' to file: {e}")
            raise

    @staticmethod
    def get_current_week_to_predict() -> str | None: 
        """
        Retrieves the current week that needs to be predicted from a file.
        Returns:
            str: The week in 'YYYYMMDD' format, or None if not found/error.
        """
        if not os.path.exists(CURRENT_PREDICTION_WEEK_FILE):
            logger.warning(f"Static prediction week file not found at {CURRENT_PREDICTION_WEEK_FILE}.")
            return None
        try:
            with open(CURRENT_PREDICTION_WEEK_FILE, 'r') as f:
                week_str = f.read().strip()
            logger.info(f"Static prediction week counter retrieved: {week_str}.")
            return week_str
        except Exception as e:
            logger.error(f"Error reading current prediction week from file {CURRENT_PREDICTION_WEEK_FILE}: {e}")
            return None