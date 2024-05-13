import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Загрузка предобработанных данных
X_train_scaled = pd.read_csv("train/build_price_train_preprocessed.csv")
y_train = pd.read_csv("train/build_price_train.csv")['price_doc']

# Обучение модели
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Сохранение обученной модели
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель успешно обучена и сохранена.")
