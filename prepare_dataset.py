import os
import pandas as pd
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)
    
    samples_per_dataset = 100000
    
    print(f"Скачивание отзывов d0rj/geo-reviews-dataset-2023 (беру {samples_per_dataset} шт.)...")
    geo_dataset = load_dataset("d0rj/geo-reviews-dataset-2023", split=f"train[:{samples_per_dataset}]")
    df_geo = geo_dataset.to_pandas()
    
    # Оставляем только нужные колонки и переименовываем
    if 'rating' in df_geo.columns and 'text' in df_geo.columns:
        df_geo = df_geo[['text', 'rating']]
    
    print(f"Скачивание отзывов nyuuzyou/wb-feedbacks (беру {samples_per_dataset} шт.)...")
    wb_dataset = load_dataset("nyuuzyou/wb-feedbacks", split=f"train[:{samples_per_dataset}]")
    df_wb = wb_dataset.to_pandas()
    
    # В датасете WB колонки могут называться иначе, приводим к стандарту
    df_wb = df_wb.rename(columns={'productValuation': 'rating'})
    if 'text' in df_wb.columns and 'rating' in df_wb.columns:
        df_wb = df_wb[['text', 'rating']]
        
    print("Объединение и обработка данных...")
    df = pd.concat([df_geo, df_wb], ignore_index=True)
    
    # Очистка пустых значений
    df = df.dropna(subset=['text', 'rating'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""]
    
    # Перевод 5-балльной шкалы в 10-балльную (1->2, 2->4, ..., 5->10)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    df['rating'] = df['rating'].astype(float)
    
    # Если максимальная оценка 5, умножаем на 2
    if df['rating'].max() <= 5.0:
        df['rating'] = df['rating'] * 2
        
    # Ограничиваем шкалу от 1 до 10 на всякий случай
    df['rating'] = df['rating'].clip(1, 10)
    
    # Перемешиваем датасет
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    save_path = "data/reviews.csv"
    # Ограничиваем каждый рейтинг максимум 13 000 случайными примерами
    df = df.groupby('rating', group_keys=False).apply(lambda x: x.sample(min(len(x), 13000), random_state=42)).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    
    print(f"Готово! Сохранено {len(df)} отзывов в {save_path}")
    print("Распределение оценок:")
    print(df['rating'].value_counts().sort_index())

if __name__ == "__main__":
    main()