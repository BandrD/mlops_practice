import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Загрузите данные из файла (например, temperature_train.csv)
train_data = pd.read_csv('train/temperature_train.csv')

# Выделите признаки (в данном случае только температура)
X_train = train_data[['Temperature']]

# Создайте и обучите StandardScaler на обучающих данных
scaler = StandardScaler().fit(X_train)

# Преобразуйте обучающие данные
X_train_scaled = scaler.transform(X_train)

# Сохраните обученный scaler в файл (например, scaler.scl)
with open('scaler.scl', 'wb') as f:
    pickle.dump(scaler, f)

print("Данные успешно предобработаны и scaler сохранен.")