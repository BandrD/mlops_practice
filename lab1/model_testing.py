import pandas as pd
import pickle

test_data = pd.read_csv('test/temperature_test.csv')

# Выделение признака
X_test = test_data[['Temperature']]

# Загрузка обученной модели
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Выполнение предсказаний на тестовых данных
y_pred = model.predict(X_test)

print("Результаты предсказаний на тестовых данных:")
print(y_pred)


print("Модель успешно протестирована на тестовых данных.")