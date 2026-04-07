import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import config
import random
from preprocessing import preprocess_batch
from model_tfidf_ridge import TFIDFRidgeModel
from model_catboost import CatBoostModel
from ensemble import EnsembleModel

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    seed_everything()

    print(f"Найден датасет: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    df = df.dropna(subset=[config.TEXT_COL, config.LABEL_COL])
    texts_raw = df[config.TEXT_COL].tolist()
    labels = df[config.LABEL_COL].values.astype(float)

    tr_raw, te_raw, tr_y, te_y = train_test_split(texts_raw, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    print("\nПредобработка тестовой выборки...")
    t0 = time.time()
    lemmatize = len(texts_raw) < 50_000
    te_prep = preprocess_batch(te_raw, lemmatize=lemmatize)
    print(f"Предобработка завершена: {time.time()-t0:.1f}с")

    results = {}

    print("\n" + "═"*55)
    print("  ЗАГРУЗКА И ОЦЕНКА МОДЕЛЕЙ")
    print("═"*55)

    m1 = TFIDFRidgeModel.load()
    p1 = m1.predict(te_prep)
    results["TF-IDF + Ridge"] = rmse(te_y, p1)

    m2 = CatBoostModel.load()
    p2 = m2.predict(te_prep, te_raw)
    results["CatBoost"] = rmse(te_y, p2)

    ens = EnsembleModel.load()
    p3 = ens.predict(te_prep, te_raw)
    results["Ансамбль"] = rmse(te_y, p3)

    print("\n" + "═"*50)
    print(f"  {'Модель':<25}  {'RMSE':>8}")
    print("─"*50)
    best = min(results.values())
    for name, score in results.items():
        print(f"  {name:<25}  {score:>8.4f}")
    print("═"*50)

    demo_texts = [
        "Превосходный товар, беру уже второй раз — качество не подвело!",
        "Сломалось на третий день. Деньги выброшены, поддержка молчит.",
        "Товар как товар. Работает, не пахнет, инструкция есть. Большего не ждал.",
        "Великолепно! Посылка шла полтора месяца, а внутри оказался совсем другой товар. Браво, так держать.",
        "Потрясающий сервис — оператор сбросил трубку всего четыре раза. Личный рекорд продавца.",
        "Грязный номер, клопы, шум с улицы до утра. Потребовал возврат денег.",
        "Брак. Выбросил сразу.",
        "Чистые номера, тихо, персонал вежливый. Больше ничего и не нужно было.",
        "Паста была божественная, шеф явно знает своё дело. Вернусь обязательно.",

    ]
    demo_prep = preprocess_batch(demo_texts, lemmatize=lemmatize)
    demo_preds = ens.predict(demo_prep, demo_texts)

    print("\nДемо предсказаний (ансамбль):")
    for text, score in zip(demo_texts, demo_preds):
        print(f"  {score:4.1f}/10 — {text[:60]}")

if __name__ == "__main__":
    main()