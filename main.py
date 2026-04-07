import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import config
from preprocessing import preprocess_batch
from model_tfidf_ridge import TFIDFRidgeModel
from model_catboost import CatBoostModel
from ensemble import EnsembleModel

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    os.makedirs("models", exist_ok=True)

    print(f"Найден датасет: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    df = df.dropna(subset=[config.TEXT_COL, config.LABEL_COL])
    texts_raw = df[config.TEXT_COL].tolist()
    labels = df[config.LABEL_COL].values.astype(float)

    print(f"Всего примеров: {len(texts_raw)}")

    tr_raw, te_raw, tr_y, te_y = train_test_split(
        texts_raw, labels, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    tr_raw_cb, val_raw, tr_y_cb, val_y = train_test_split(
        tr_raw, tr_y, test_size=0.1,
        random_state=config.RANDOM_STATE,
    )

    print("\nПредобработка...")
    t0 = time.time()
    lemmatize = len(texts_raw) < 50_000
    tr_prep = preprocess_batch(tr_raw_cb, lemmatize=lemmatize)
    val_prep = preprocess_batch(val_raw, lemmatize=lemmatize)
    te_prep = preprocess_batch(te_raw, lemmatize=lemmatize)
    tr_prep_full = preprocess_batch(tr_raw, lemmatize=lemmatize)
    print(f"Предобработка: {time.time()-t0:.1f}с")

    results = {}

    print("\n" + "═"*55)
    print("  МОДЕЛЬ 1: TF-IDF + Ridge")
    print("═"*55)
    m1 = TFIDFRidgeModel()
    m1.fit(tr_prep_full, tr_y)
    p1 = m1.predict(te_prep)
    results["TF-IDF + Ridge"] = rmse(te_y, p1)
    m1.save()

    print("\n" + "═"*55)
    print("  МОДЕЛЬ 2: CatBoost")
    print("═"*55)
    m2 = CatBoostModel()
    m2.fit(tr_prep, tr_raw_cb, tr_y_cb,
           eval_texts_preprocessed=val_prep,
           eval_texts_raw=val_raw,
           eval_labels=val_y)
    p2 = m2.predict(te_prep, te_raw)
    results["CatBoost"] = rmse(te_y, p2)
    m2.save()

    print("\n" + "═"*55)
    print("  МОДЕЛЬ 3: Ансамбль")
    print("═"*55)
    ens = EnsembleModel()
    ens.ridge = m1
    ens.catboost = m2
    ens.optimize_weights(val_prep, val_raw, val_y)
    p3 = ens.predict(te_prep, te_raw)
    results["Ансамбль"] = rmse(te_y, p3)

    print("\n" + "═"*50)
    print(f"  {'Модель':<25}  {'RMSE':>8}")
    print("─"*50)
    best = min(results.values())
    for name, score in results.items():
        marker = " ◄" if score == best else ""
        print(f"  {name:<25}  {score:>8.4f}{marker}")
    print("═"*50)

    demo_texts = [
        "Превосходный товар, очень доволен! Однозначно рекомендую.",
        "Плохое качество, разочарован покупкой. Вернул обратно.",
        "Обычный товар, ничего особенного. Средненько.",
        "Быстрая доставка, хорошая упаковка. В целом доволен.",
    ]
    demo_prep = preprocess_batch(demo_texts, lemmatize=lemmatize)
    demo_preds = ens.predict(demo_prep, demo_texts)

    print("\nДемо предсказаний (ансамбль):")
    for text, score in zip(demo_texts, demo_preds):
        print(f"  {score:4.1f}/10 — {text[:60]}")

    print(f"\nДля предсказания: from predict import predict, print_predictions")
    print(f"Для обучения:     from train import main as train; train('data/reviews.csv')")


if __name__ == "__main__":
    main()