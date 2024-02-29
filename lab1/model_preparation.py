import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Загрузите обучающие данные (например, temperature_train.csv)
train_data = pd.read_csv('train/temperature_train.csv')

# Выделите признаки (в данном случае только температура)
X_train = train_data[['Temperature']]

# Загрузите целевую переменную (например, цена)
y_train = train_data['Temperature']  # Замените 'Target' на имя вашей целевой переменной

# Создайте и обучите модель (например, линейную регрессию)
model = LinearRegression()
model.fit(X_train, y_train)

# Сохраните обученную модель в файл (например, model.pkl)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель успешно обучена и сохранена.")