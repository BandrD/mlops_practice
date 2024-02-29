import pandas as pd
import pickle

# Загрузите тестовые данные (например, temperature_test.csv)
test_data = pd.read_csv('test/temperature_test.csv')

# Выделите признаки (в данном случае только температура)
X_test = test_data[['Temperature']]

# Загрузите обученную модель из файла (например, model.pkl)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Выполните предсказания на тестовых данных
y_pred = model.predict(X_test)

# Выведите результаты предсказаний (например, среднеквадратичная ошибка)
print("Результаты предсказаний на тестовых данных:")
print(y_pred)

# Дополнительно можно оценить метрики качества модели (например, R²)
# Например:
# from sklearn.metrics import r2_score
# r2 = r2_score(y_true, y_pred)
# print(f"R²: {r2}")

print("Модель успешно протестирована на тестовых данных.")