import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Загрузка обучающих данных
train_data = pd.read_csv('train/temperature_train.csv')

# Выделение признака
X_train = train_data[['Temperature']]

# Целевая переменая
y_train = train_data['Temperature'] 

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель успешно обучена и сохранена.")
