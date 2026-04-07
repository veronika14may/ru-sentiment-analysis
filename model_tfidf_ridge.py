import numpy as np
import joblib
from sklearn.linear_model import Ridge
import config
from features import TFIDFFeatureExtractor

class TFIDFRidgeModel:
    def __init__(self, alpha=config.RIDGE_ALPHA):
        self.extractor = TFIDFFeatureExtractor()
        self.ridge = Ridge(alpha=alpha)

    def fit(self, texts, labels):
        print("[TFIDFRidge] Извлечение признаков...")
        X = self.extractor.fit_transform(texts)
        print(f"[TFIDFRidge] Матрица признаков: {X.shape}")
        print("[TFIDFRidge] Обучение Ridge...")
        self.ridge.fit(X, labels)
        print("[TFIDFRidge] Готово.")
        return self

    def predict(self, texts):
        X = self.extractor.transform(texts)
        preds = self.ridge.predict(X)
        return np.clip(preds, 1.0, 10.0)

    def save(self, path=config.TFIDF_RIDGE_PATH):
        joblib.dump(self, path)
        print(f"[TFIDFRidge] Модель сохранена {path}")

    @staticmethod
    def load(path=config.TFIDF_RIDGE_PATH):
        model = joblib.load(path)
        print(f"[TFIDFRidge] Загружена из {path}")
        return model