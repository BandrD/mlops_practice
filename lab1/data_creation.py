import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Загрузка данных
data = pd.read_csv("build_price.csv")

# Создание папок "train" и "test", если они не существуют
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Удаление столбцов 'timestamp' и 'sub_area'
build_price = data.drop(columns=['timestamp', 'sub_area'])

# Разделение данных на обучающий и тестовый наборы
train_data, test_data = train_test_split(build_price, test_size=0.2, random_state=42)

# Сохранение обучающего и тестового наборов данных
train_data.to_csv('train/build_price_train.csv', index=False)
test_data.to_csv('test/build_price_test.csv', index=False)


print("Данные успешно созданы и сохранены в папках 'train' и 'test'.")
