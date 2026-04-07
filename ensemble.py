import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
from model_tfidf_ridge import TFIDFRidgeModel
from model_catboost import CatBoostModel
import config

class EnsembleModel:
    def __init__(self, weight_ridge=config.ENSEMBLE_WEIGHT_RIDGE, weight_catboost=config.ENSEMBLE_WEIGHT_CATBOOST):
        self.w_ridge = weight_ridge
        self.w_catboost = weight_catboost
        self.ridge = None
        self.catboost = None

    def fit(self, texts_preprocessed, texts_raw, labels, val_texts_prep=None, val_texts_raw=None, val_labels=None):
        print("\n" + "="*50)
        print("--- Обучение TF-IDF + Ridge ---")
        print("="*50)
        self.ridge = TFIDFRidgeModel()
        self.ridge.fit(texts_preprocessed, labels)
        print("\n" + "="*50)
        print("--- Обучение CatBoost ---")
        print("="*50)
        self.catboost = CatBoostModel()
        self.catboost.fit(texts_preprocessed, texts_raw, labels, eval_texts_preprocessed=val_texts_prep, eval_texts_raw=val_texts_raw, eval_labels=val_labels)
        if val_texts_prep is not None:
            self.optimize_weights(val_texts_prep, val_texts_raw, val_labels)
        return self

    def optimize_weights(self, texts_prep, texts_raw, labels):
        print("\n[Ensemble] Подбор весов на валидации...")
        p1 = self.ridge.predict(texts_prep)
        p2 = self.catboost.predict(texts_prep, texts_raw)
        def objective(w):
            blend = np.clip(w * p1 + (1 - w) * p2, 1.0, 10.0)
            return np.sqrt(mean_squared_error(labels, blend))
        res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        self.w_ridge = float(res.x)
        self.w_catboost = 1.0 - self.w_ridge
        print(f"[Ensemble] Оптимальные веса: Ridge={self.w_ridge:.3f}, CatBoost={self.w_catboost:.3f}  (RMSE val={res.fun:.4f})")

    def predict(self, texts_preprocessed, texts_raw):
        p1 = self.ridge.predict(texts_preprocessed)
        p2 = self.catboost.predict(texts_preprocessed, texts_raw)
        blend = self.w_ridge * p1 + self.w_catboost * p2
        return np.clip(blend, 1.0, 10.0)

    def save(self):
        self.ridge.save()
        self.catboost.save()

    @staticmethod
    def load(weight_ridge=config.ENSEMBLE_WEIGHT_RIDGE):
        obj = EnsembleModel.__new__(EnsembleModel)
        obj.ridge = TFIDFRidgeModel.load()
        obj.catboost = CatBoostModel.load()
        obj.w_ridge = weight_ridge
        obj.w_catboost = 1.0 - weight_ridge
        return obj