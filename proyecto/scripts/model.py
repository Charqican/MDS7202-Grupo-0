from typing import List, Tuple
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import tqdm
import coloredlogs, logging



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


    def get_test_batch(self, week:int):
        return self.data[self.data[self.col]==week]
    

    def __len__(self): return len(self.weeks_batches)
    def __getitem__(self,index:int): return self.get_train_eval_batch(index)
    def __iter__(self):
        for i in range(len(self)): yield self.get_train_eval_batch(i)



class IncrementalXGBoost(BaseEstimator, ClassifierMixin):

    Recursive = 0
    logger = logging.getLogger(__name__) 
    coloredlogs.install(level='DEBUG', logger=logger)

    def __init__(self, batch_col='batch', label_col='label', f1_threshold=0.6,
                reset_window_size=4, initial_lr=0.01, final_lr=0.1, max_depth=6,
                n_estimators=100, log = True):

        self.batch_col = batch_col
        self.label_col = label_col
        self.history_ = []
        self.f1_threshold = f1_threshold
        self.reset_window_size = reset_window_size
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model_ = None
        self.log = log
        self.eval_scores = []
        self.f1_scores = []
        
    
    def fit(self, X, y):
        

        self.history_ = X
        total_batches = len(X)
        reset_flag = False
        model = self.model
        eval_scores = []
        f1_scores = []
        for i, (train_df, eval_df) in tqdm.notebook.tqdm(enumerate(X, start=1)):

            if self.log: IncrementalXGBoost.logger.info(f"\n Entrenando batch {i}/{total_batches}")


            # Data drift
            if reset_flag and IncrementalXGBoost.Recursive < 3:
                model = None
                if self.log: IncrementalXGBoost.logger.info(f"Reset activado. Usando ventana de entrenamiento de los últimos {self.reset_window_size} batches.")
                IncrementalXGBoost.Recursive += 1
                recent_batches = self.history_[max(0, i - self.reset_window_size):i]
                self.fit(recent_batches)
                reset_flag = False
            
            elif IncrementalXGBoost.Recursive >= 3: 
                IncrementalXGBoost.Recursive = 0
                return

            X_train, y_train = train_df.drop(columns=[self.label_col]), train_df[self.label_col]
            X_eval, y_eval = eval_df.drop(columns=[self.label_col]), eval_df[self.label_col]
            
            # weighted class batches
            scale_pos = IncrementalXGBoost.scalePosWeight(y_train)
            if self.log: IncrementalXGBoost.logger.info(f"scale_pos_weight = {scale_pos:.2f}")

            
            # Entrenamiento (nuevo o incremental)
            if model is None:
                model = XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.1,
                    eval_metric='auc',
                    objective='binary:logistic',
                    tree_method='auto',
                    scale_pos_weight=scale_pos,
                    early_stopping_rounds=12,
                )
                model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=0)

            else:
                # Entrenamos sobre el anterior.
                previous_model = model
                model = XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.1,
                    eval_metric='auc',
                    objective='binary:logistic',
                    tree_method='auto',
                    scale_pos_weight=scale_pos,
                    early_stopping_rounds=12,
                )
                model.fit(
                    X_train, y_train,
                    xgb_model=previous_model.get_booster(),
                    eval_set=[(X_eval, y_eval)],
                    verbose=0
                )
            
            # Evaluación
            y_pred_probs = model.predict_proba(X_eval)[:, 1]
            y_pred_labels = (y_pred_probs >= 0.5).astype(int)
    
            if len(np.unique(y_eval)) < 2:
                if self.log : IncrementalXGBoost.logger.warning("Batch {i} con clase única. AUC y F1 no disponibles.")
                auc, f1 = np.nan, np.nan
            else:
                auc = roc_auc_score(y_eval, y_pred_probs)
                f1 = f1_score(y_eval, y_pred_labels)
                if self.log : IncrementalXGBoost.logger.info(f"AUC: {auc:.4f} | F1-score: {f1:.4f}")
    
            eval_scores.append(auc)
            f1_scores.append(f1)
    
            # Chequeo de reset
            if shouldReset(f1_scores[i], f1_scores[i-1]):
                if self.log : IncrementalXGBoost.logger.warning(f"F1-score {f1:.4f} por debajo del umbral. Se programará un reset.")
                reset_flag = True
    

    @staticmethod
    def shouldReset(f1_score : float, last_f1_score : float): 
        if (last_f1_score - f1_score) > .1 :
            return True
        else:
            return False
    

    def flushHistory(self):
        self.history_ = []

    def scalePosWeight(self, y_train) -> float:
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
        return scale_pos


    def partial_fit(batches : Batch):
        ...



def incremental_xgboost_training(batches: List[Tuple[pd.DataFrame, pd.DataFrame]], label_col: str = 'label',
                                f1_threshold: float = 0.3, reset_window_size: int = 2, initial_lr: float = 0.01,
                                final_lr: float = 0.1):
    
    model = None
    eval_scores = []
    f1_scores = []
    resets = []
    total_batches = len(batches)
    reset_flag = False

    for i, (train_df, eval_df) in tqdm.notebook.tqdm(enumerate(batches, start=1)):
        print(f"\n Entrenando batch {i}/{total_batches}")

        if reset_flag:
            print(f"🔁 Reset activado. Usando ventana de entrenamiento de los últimos {reset_window_size} batches.")
            recent_batches = batches[max(0, i - reset_window_size):i]
            train_df = pd.concat([b[0] for b in recent_batches])
            reset_flag = False
            model = None  # Forzamos reinicialización

        X_train, y_train = train_df.drop(columns=[label_col]), train_df[label_col]
        X_eval, y_eval = eval_df.drop(columns=[label_col]), eval_df[label_col]

        # ⚖️ Ajuste de clases
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"⚖️ scale_pos_weight = {scale_pos:.2f}")

        # Entrenamiento (nuevo o incremental)
        if model is None:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss',
                objective='binary:logistic',
                tree_method='auto',
                scale_pos_weight=scale_pos,
                early_stopping_rounds=15,
            )
            model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=0)
        else:
            previous_model = model
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss',
                objective='binary:logistic',
                tree_method='auto',
                scale_pos_weight=scale_pos,
                early_stopping_rounds=15,
            )
            model.fit(
                X_train, y_train,
                xgb_model=previous_model.get_booster(),
                eval_set=[(X_eval, y_eval)],
                verbose=0
            )

        # Evaluación
        y_pred_probs = model.predict_proba(X_eval)[:, 1]
        y_pred_labels = (y_pred_probs >= 0.5).astype(int)

        if len(np.unique(y_eval)) < 2:
            print(f"⚠️ Batch {i} con clase única. AUC y F1 no disponibles.")
            auc, f1 = np.nan, np.nan
        else:
            auc = roc_auc_score(y_eval, y_pred_probs)
            f1 = f1_score(y_eval, y_pred_labels)
            print(f"✅ AUC: {auc:.4f} | F1-score: {f1:.4f}")

        eval_scores.append(auc)
        f1_scores.append(f1)

        # Chequeo de reset
        if not np.isnan(f1) and f1 < f1_threshold:
            print(f"🔻 F1-score {f1:.4f} por debajo del umbral {f1_threshold:.2f}. Se programará un reset.")
            reset_flag = True
            resets.append(i)
        else:
            resets.append(None)

    return model, eval_scores, f1_scores, resets