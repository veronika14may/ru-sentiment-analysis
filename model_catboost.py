import numpy as np
import joblib
from catboost import CatBoostRegressor
import config
from features import SVDFeatureExtractor

class CatBoostModel:
    def __init__(self):
        self.extractor = SVDFeatureExtractor()
        self.model = CatBoostRegressor(
            iterations=config.CB_ITERATIONS,
            learning_rate=config.CB_LEARNING_RATE,
            depth=config.CB_DEPTH,
            loss_function=config.CB_LOSS,
            eval_metric=config.CB_EVAL_METRIC,
            random_seed=config.RANDOM_STATE,
            early_stopping_rounds=50,
            verbose=100,
        )

    def fit(self, texts_preprocessed, texts_raw, labels, eval_texts_preprocessed=None, eval_texts_raw=None, eval_labels=None):
        print("[CatBoost] Извлечение признаков...")
        X_train = self.extractor.fit_transform(texts_preprocessed, texts_raw)
        print(f"[CatBoost] Матрица признаков: {X_train.shape}")
        
        eval_set = None
        if eval_texts_preprocessed is not None:
            X_eval = self.extractor.transform(eval_texts_preprocessed, eval_texts_raw)
            eval_set = (X_eval, eval_labels)
            
        print("[CatBoost] Обучение...")
        self.model.fit(
            X_train, labels,
            eval_set=eval_set,
            use_best_model=(eval_set is not None),
        )
        print("[CatBoost] Готово.")
        return self

    def predict(self, texts_preprocessed, texts_raw):
        X = self.extractor.transform(texts_preprocessed, texts_raw)
        preds = self.model.predict(X)
        return np.clip(preds, 1.0, 10.0)

    def save(self):
        self.model.save_model(config.CATBOOST_MODEL_PATH)
        self.extractor.save(config.CATBOOST_EXTR_PATH)
        print(f"[CatBoost] Модель сохранена {config.CATBOOST_MODEL_PATH}")

    @staticmethod
    def load():
        obj = object.__new__(CatBoostModel)
        obj.model = CatBoostRegressor()
        obj.model.load_model(config.CATBOOST_MODEL_PATH)
        obj.extractor = SVDFeatureExtractor.load(config.CATBOOST_EXTR_PATH)
        print(f"[CatBoost] Загружена из {config.CATBOOST_MODEL_PATH}")
        return obj