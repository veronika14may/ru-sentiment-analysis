import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import config
from preprocessing import preprocess_batch
from model_tfidf_ridge import TFIDFRidgeModel
from model_catboost import CatBoostModel
from ensemble import EnsembleModel

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def load_data(path):
    print(f"Загрузка данных: {path}")
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=[config.TEXT_COL, config.LABEL_COL])
    df[config.LABEL_COL] = pd.to_numeric(df[config.LABEL_COL], errors="coerce")
    df = df.dropna(subset=[config.LABEL_COL])
    df = df[df[config.LABEL_COL].between(1, 10)]
    df[config.TEXT_COL] = df[config.TEXT_COL].astype(str)
    print(f"Строк после очистки: {len(df)} / {before}")
    print("Распределение оценок:")
    dist = df[config.LABEL_COL].value_counts().sort_index()
    for score, cnt in dist.items():
        bar = "█" * int(cnt / dist.max() * 30)
        print(f"  {int(score):2d}: {bar} {cnt}")
    return df[config.TEXT_COL].tolist(), df[config.LABEL_COL].values.astype(float)

def print_results(results):
    print("\n" + "═"*45)
    print(f"{'Модель':<25}{'RMSE':>10}")
    print("─"*45)
    best = min(results.values())
    for name, score in results.items():
        print(f"  {name:<23}{score:>8.4f}")
    print("═"*45)
    print("(RMSE: чем меньше, тем лучше)\n")

def main(data_path=config.DATA_PATH, lemmatize=True, alpha=config.RIDGE_ALPHA):
    os.makedirs("models", exist_ok=True)
    texts_raw, labels = load_data(data_path)
    tr_raw, te_raw, tr_labels, te_labels = train_test_split(texts_raw, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    print(f"\nTrain: {len(tr_raw)}, Test: {len(te_raw)}")
    tr_raw_cb, val_raw, tr_labels_cb, val_labels = train_test_split(tr_raw, tr_labels, test_size=0.1, random_state=config.RANDOM_STATE)
    print(f"\nПредобработка (лемматизация={'включена' if lemmatize else 'выключена'})...")
    t0 = time.time()
    tr_prep = preprocess_batch(tr_raw_cb, lemmatize=lemmatize)
    val_prep = preprocess_batch(val_raw, lemmatize=lemmatize)
    te_prep = preprocess_batch(te_raw, lemmatize=lemmatize)
    tr_prep_full = preprocess_batch(tr_raw, lemmatize=lemmatize)
    print(f"Предобработка завершена за {time.time()-t0:.1f}с")
    results = {}
    
    print("\n" + "="*50)
    print("=== Модель 1: TF-IDF + Ridge ===")
    print("="*50)
    ridge_model = TFIDFRidgeModel(alpha=alpha)
    ridge_model.fit(tr_prep_full, tr_labels)
    ridge_preds = ridge_model.predict(te_prep)
    results["TF-IDF + Ridge"] = rmse(te_labels, ridge_preds)
    ridge_model.save()
    
    print("\n" + "="*50)
    print("=== Модель 2: CatBoost ===")
    print("="*50)
    cb_model = CatBoostModel()
    cb_model.fit(tr_prep, tr_raw_cb, tr_labels_cb, eval_texts_preprocessed=val_prep, eval_texts_raw=val_raw, eval_labels=val_labels)
    cb_preds = cb_model.predict(te_prep, te_raw)
    results["CatBoost"] = rmse(te_labels, cb_preds)
    cb_model.save()
    
    print("\n" + "="*50)
    print("=== Модель 3: Ансамбль (взвешенное среднее) ===")
    print("="*50)
    ens = EnsembleModel()
    ens.ridge = ridge_model
    ens.catboost = cb_model
    ens.optimize_weights(val_prep, val_raw, val_labels)
    ens_preds = ens.predict(te_prep, te_raw)
    results["Ансамбль"] = rmse(te_labels, ens_preds)
    
    print_results(results)
    
    print("Примеры предсказаний (ансамбль):")
    for i in range(min(5, len(te_raw))):
        preview = te_raw[i][:70].replace("\n", " ")
        print(f"  [{ens_preds[i]:4.1f} / реально {te_labels[i]:.0f}] {preview}")

if __name__ == "__main__":
    main()