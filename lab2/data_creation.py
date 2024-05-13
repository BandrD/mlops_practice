import pandas as pd
from sklearn.model_selection import train_test_split
import os
import requests
import io

# Загрузка данных
url = 'https://raw.github.com/BandrD/mlops_practice/raw/main/lab1/train/temperature_train.csv'
response = requests.get(url)
    
if response.status_code == 200:
    # Загрузка данных в объект DataFrame
    data = pd.read_csv(io.StringIO(response.text))
    print("Данные успешно загружены.")
else:
    print("Ошибка загрузки данных:", response.status_code)

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
