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
FEATURES_DIR = './data/processed/'
LABELS_DIR = './data/processed/labels/'
CURRENT_PREDICTION_WEEK_FILE = './data/processed/model_state/current_prediction_week.txt'
PREDICTIONS_DIR = './data/processed/predictions/' 

class IncrementalXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, f1_threshold_drop: float = 0.05,
                 reset_window_size: int = 8, 
                 initial_training_weeks: int = 8, 
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
        self.current_week_to_predict = 0


    def _log(self, level, message):
        logger.log(level, f"[{self.__class__.__name__}] {message}")

    def _create_or_update_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_eval: pd.DataFrame = None, y_eval: pd.Series = None, 
                                is_full_retrain: bool = False) -> XGBClassifier:
        
        self._log(logging.INFO, f"Initiating XGBoost training/update. Train shape: {X_train.shape}, Eval shape: {X_eval.shape}")
        
        fit_params = {
            'eval_set': [(X_eval, y_eval)],
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

    def initial_train(self):
        self._log(logging.INFO, "Starting cold start training.")

        features_path = os.path.join(FEATURES_DIR, 'transformed_enriched.parquet')
        features = pd.read_parquet(features_path)

        all_label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.startswith('labels_week_') and f.endswith('.parquet')])

        if len(all_label_files) < self.initial_training_weeks:
            self._log(logging.ERROR, f"Insufficient label weeks for cold start. Found {len(all_label_files)}, required {self.initial_training_weeks}.")
            raise ValueError("Insufficient label data for cold start.")

        initial_dfs = []
        for i in range(self.initial_training_weeks):
            label_path = os.path.join(LABELS_DIR, all_label_files[i])
            labels = pd.read_parquet(label_path)

            week_str = all_label_files[i].replace('labels_week_', '').replace('.parquet', '')
            features_with_week = features.copy()
            features_with_week['week_num'] = int(week_str)

            merged_df = pd.merge(features_with_week, labels, on=['customer_id', 'product_id', 'week_num'], how='left')
            merged_df[self.label_col] = merged_df[self.label_col].fillna(0).astype(int)

            initial_dfs.append(merged_df)
            self.history_.append(merged_df)

        combined_df = pd.concat(initial_dfs, ignore_index=True)
        X_train_initial = combined_df.drop(columns=[self.label_col], errors='ignore')
        y_train_initial = combined_df[self.label_col]

        self._create_or_update_model(X_train_initial, y_train_initial, X_train_initial.copy(), y_train_initial.copy(), is_full_retrain=True)
        self._log(logging.INFO, "Cold start training completed.")

        
        last_trained_week_str = all_label_files[self.initial_training_weeks - 1].replace('labels_week_', '').replace('.parquet', '')
        last_trained_week_num = int(last_trained_week_str)
        self.current_week_to_predict = last_trained_week_num + 1  # Suma 1 para apuntar a la próxima semana
        self._log(logging.INFO, f"Next week for prediction initialized to: {self.current_week_to_predict}.")


    def generate_predictions_for_next_week(self) -> pd.DataFrame:
        prediction_week = self.current_week_to_predict
        if prediction_week is None:
            self._log(logging.ERROR, "Current prediction week is not set. Cannot generate predictions.")
            raise RuntimeError("Current prediction week is not set. Run initial_train first or set it manually.")

        self._log(logging.INFO, f"Generating predictions for Week {prediction_week}.")

        if self.model is None:
            self._log(logging.ERROR, "Model not trained. Cannot make predictions.")
            raise RuntimeError("Model is not trained. Please train it first.")

        transformed_path = os.path.join(FEATURES_DIR, 'transformed_enriched.parquet')
        if not os.path.exists(transformed_path):
            self._log(logging.ERROR, f"Transformed features file not found at {transformed_path}.")
            raise FileNotFoundError(f"Transformed features file not found at {transformed_path}.")

        features_df = pd.read_parquet(transformed_path).copy()

        # Agregar la semana actual para predicción
        features_df['week_num'] = int(prediction_week)

        # Preparar features para el modelo
        X_predict = features_df.drop(columns=['customer_id', 'product_id'], errors='ignore')
        if self.label_col in X_predict.columns:
            X_predict = X_predict.drop(columns=[self.label_col])

        probabilities = self.model.predict_proba(X_predict)[:, 1]

        predictions_df = features_df[['customer_id', 'product_id', 'week_num']].copy()
        predictions_df['prediction_proba'] = probabilities

        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        prediction_output_path = os.path.join(PREDICTIONS_DIR, f"predictions_week_{prediction_week}.parquet")
        predictions_df.to_parquet(prediction_output_path, index=False)
        self._log(logging.INFO, f"Predictions for Week {prediction_week} saved to {prediction_output_path}.")
        self._log(logging.INFO, f"Predictions generated for {len(predictions_df)} customer-product pairs for Week {prediction_week}.")

        return predictions_df


    def _evaluate_performance(self, model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
        if self.last_evaluated_f1 is None:
            self._log(logging.INFO, "Previous F1 is None - cold start -. Skipping retraining logic.")
            # En cold start no avanzamos la semana automáticamente aquí, porque no tenemos contexto.
            # Se asume que esto se maneja fuera, o en initial_train.
            return None, None  # o (0.0, 0.0) si prefieres valores numéricos

        if X.empty or y.empty:
            self._log(logging.WARNING, "No data for performance evaluation. Returning 0.0 for AUC and F1-score.")
            return 0.0, 0.0

        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc_score = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)

        self._log(logging.INFO, f"Evaluation results: AUC={auc_score:.4f}, F1-score={f1:.4f}")
        return auc_score, f1


    # --- MÉTODOS PARA EL FLUJO DE TRABAJO DEL DAG ---
    def train_incremental_and_update_history(self, week_t_date_str: str):
        self._log(logging.INFO, f"Starting incremental training for Week {week_t_date_str}.")

        features_path = os.path.join(FEATURES_DIR, 'transformed_enriched.parquet')  # path fijo al dataset transformado
        labels_path = os.path.join(LABELS_DIR, f"labels_week_{week_t_date_str}.parquet")

        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            self._log(logging.ERROR, f"Missing features or labels for Week {week_t_date_str} at {features_path} or {labels_path}.")
            raise FileNotFoundError(f"Data for Week {week_t_date_str} not found for training.")

        features_t = pd.read_parquet(features_path)
        labels_t = pd.read_parquet(labels_path)

        # Agregamos la columna week_num para esta semana en features antes del merge
        features_t = features_t.copy()
        features_t['week_num'] = int(week_t_date_str)

        full_week_t_df = pd.merge(features_t, labels_t, on=['customer_id', 'product_id', 'week_num'], how='left')

        if self.xgb_params['objective'] == 'binary:logistic':
            full_week_t_df[self.label_col] = full_week_t_df[self.label_col].fillna(0).astype(int)

        self.history_.append(full_week_t_df)

        X_train = full_week_t_df.drop(columns=[self.label_col, 'customer_id', 'product_id'], errors='ignore')
        y_train = full_week_t_df[self.label_col]

        X_eval = X_train.copy()
        y_eval = y_train.copy()

        self._create_or_update_model(X_train, y_train, X_eval, y_eval, is_full_retrain=False)
        self._log(logging.INFO, f"Incremental training for Week {week_t_date_str} completed. Model updated.")





    def evaluate_and_detect_drift(self):
        """
        Evaluates predictions for the 'current prediction week' once actual labels are available.
        This method is designed to be run AFTER generate_predictions_for_next_week and when labels are ready.
        It also handles data drift detection and updates the internal current_week_to_predict counter.
        Returns:
            float | None: F1-score of the evaluated predictions, or None if data not ready.
        """
        week_to_evaluate = self.current_week_to_predict
        if week_to_evaluate is None:
            self._log(logging.ERROR, "No current week set to evaluate.")
            raise RuntimeError("Current week to evaluate is None.")

        self._log(logging.INFO, f"Starting evaluation and drift detection for Week {week_to_evaluate}.")

        predictions_path = os.path.join(PREDICTIONS_DIR, f"predictions_week_{week_to_evaluate}.parquet")
        labels_path = os.path.join(LABELS_DIR, f"labels_week_{week_to_evaluate}.parquet")

        if not os.path.exists(predictions_path) or not os.path.exists(labels_path):
            self._log(logging.INFO, f"Data (predictions or labels) for Week {week_to_evaluate} not yet fully available.")
            # Pause point for DAG, no data to evaluate yet
            return None

        predictions_df = pd.read_parquet(predictions_path)
        actual_labels_df = pd.read_parquet(labels_path)

        if self.xgb_params['objective'] == 'binary:logistic':
            actual_labels_df[self.label_col] = actual_labels_df[self.label_col].fillna(0).astype(int)

        eval_df = pd.merge(predictions_df, actual_labels_df, on=['customer_id', 'product_id', 'week_num'], how='left')
        eval_df.dropna(subset=[self.label_col, 'prediction_proba'], inplace=True)

        if eval_df.empty:
            self._log(logging.WARNING, f"No valid data points for evaluation for Week {week_to_evaluate} after merging predictions and labels. F1-score set to 0.0.")
            self.last_evaluated_f1 = 0.0
        else:
            y_true = eval_df[self.label_col]
            y_pred_proba = eval_df['prediction_proba']
            f1 = f1_score(y_true, (y_pred_proba > 0.5).astype(int))
            auc_score = roc_auc_score(y_true, y_pred_proba)
            self._log(logging.INFO, f"Evaluation for Week {week_to_evaluate} completed. AUC={auc_score:.4f}, F1-score={f1:.4f}")
            self.last_evaluated_f1 = f1

        # --- Actualizar el contador interno para la próxima semana de predicción ---
        next_prediction_week = week_to_evaluate + 1
        self.current_week_to_predict = next_prediction_week
        self._log(logging.INFO, f"Updated current week to predict to: {next_prediction_week}.")

        return self.last_evaluated_f1


    def full_retrain(self):
        """
        Performs a full retraining of the model using the last window of historical data.
        Triggered when data drift is detected or as a scheduled full refresh.
        """
        self._log(logging.INFO, "Starting full model retraining due to detected data drift or scheduled refresh.")

        if not self.history_:
            self._log(logging.ERROR, "Cannot perform full retraining: model history is empty.")
            raise ValueError("Model history is empty for full retraining.")

        combined_history_df = pd.concat(list(self.history_), ignore_index=True)

        X_train_full = combined_history_df.drop(columns=[self.label_col, 'customer_id', 'product_id'], errors='ignore')
        y_train_full = combined_history_df[self.label_col]

        X_eval_full = X_train_full.copy()
        y_eval_full = y_train_full.copy()

        self._create_or_update_model(X_train_full, y_train_full, X_eval_full, y_eval_full, is_full_retrain=True)

        _, self.last_evaluated_f1 = self._evaluate_performance(self.model, X_eval_full, y_eval_full)
        self._log(logging.INFO, f"Full retraining completed. New F1-score after retraining: {self.last_evaluated_f1:.4f}")
        self._log(logging.INFO, f"Current week to predict remains at: {self.current_week_to_predict}")

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
    
    
def set_current_week_to_predict(self, week: int):
    self.current_week_to_predict = week
    self._log(logging.INFO, f"Current week to predict set internally: {week}")

def get_current_week_to_predict(self) -> int:
    return self.current_week_to_predict