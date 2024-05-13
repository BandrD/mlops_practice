import pandas as pd
import pickle

# Загрузка тестовых данных
test_data = pd.read_csv("test/build_price_test.csv")

# Загрузка обученной модели
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Предсказание на тестовых данных
X_test = test_data.drop(columns=['price_doc'])
y_test = test_data['price_doc']
y_pred = model.predict(X_test)

# Оценка качества модели (например, средняя абсолютная ошибка)
mae = abs(y_pred - y_test).mean()
print("Mean Absolute Error:", mae)