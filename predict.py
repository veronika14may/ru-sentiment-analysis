import numpy as np
from preprocessing import preprocess_batch
from model_tfidf_ridge import TFIDFRidgeModel
from model_catboost import CatBoostModel
from ensemble import EnsembleModel

def score_to_label(score):
    if score <= 2.5: return "Очень негативный"
    if score <= 4.0: return "Негативный"
    if score <= 5.5: return "Нейтральный"
    if score <= 7.5: return "Позитивный"
    return "Очень позитивный"

def score_bar(score, width=20):
    filled = int(round((score - 1) / 9 * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"

def predict(texts, model_name="ensemble", lemmatize=True):
    texts_prep = preprocess_batch(texts, lemmatize=lemmatize)
    if model_name == "ridge":
        return TFIDFRidgeModel.load().predict(texts_prep)
    if model_name == "catboost":
        return CatBoostModel.load().predict(texts_prep, texts)
    if model_name == "ensemble":
        return EnsembleModel.load().predict(texts_prep, texts)
    raise ValueError(f"Неизвестная модель: {model_name!r}. Доступно: ridge, catboost, ensemble")

def print_predictions(texts, scores):
    print()
    for text, score in zip(texts, scores):
        preview = text[:55].replace("\n", " ") + ("…" if len(text) > 55 else "")
        print(f"  {score:4.1f}/10  {score_bar(score)}  {score_to_label(score)}")
        print(f"          {preview}")
        print()