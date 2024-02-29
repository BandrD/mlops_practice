import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Загрузика данных из файла
train_data = pd.read_csv('train/temperature_train.csv')

# Выделение признака
X_train = train_data[['Temperature']]

# Создание и обучение StandardScaler на обучающих данных
scaler = StandardScaler().fit(X_train)

# Преобразуйте обучающих данных
X_train_scaled = scaler.transform(X_train)

with open('scaler.scl', 'wb') as f:
    pickle.dump(scaler, f)

print("Данные успешно предобработаны и scaler сохранен.")