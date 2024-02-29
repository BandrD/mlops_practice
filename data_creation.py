import numpy as np
import pandas as pd
import os

# Создаем временной ряд с шумом
np.random.seed(42)
days = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
temperature = np.sin(np.arange(len(days)) / 30) * 10 + np.random.normal(0, 2, len(days))

# Создаем DataFrame с данными
df = pd.DataFrame({'Date': days, 'Temperature': temperature})

# Разделяем данные на train и test
train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Создаем папки для сохранения данных
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

# Сохраняем данные в CSV-файлах
train_data.to_csv('train/temperature_train.csv', index=False)
test_data.to_csv('test/temperature_test.csv', index=False)

print("Данные успешно созданы и сохранены в папках 'train' и 'test'.")