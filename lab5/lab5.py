import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pytest

# Фиксируем случайное состояние для воспроизводимости результатов
np.random.seed(40)

# Генерируем данные для тренировочного датасета
n_samples = 1000
area_train = np.random.normal(50, 20, n_samples)  # площадь квартиры
rooms_train = np.random.randint(1, 7, n_samples)  # количество комнат
age_train = np.random.randint(1, 50, n_samples)  # возраст дома
price_train = 500000 + 70000 * area_train + 100000 * rooms_train - 20000 * age_train + np.random.normal(0, 10000,
                                                                                                        n_samples)
# Генерируем данные для тестового датасета без изменений
area_test = np.random.normal(50, 20, n_samples)
rooms_test = np.random.randint(1, 7, n_samples)
age_test = np.random.randint(1, 50, n_samples)
price_test = 500000 + 70000 * area_test + 100000 * rooms_test - 20000 * age_test + np.random.normal(0, 10000, n_samples)

# Генерируем данные для тестового датасета с добавлением шума
price_test_noisy = price_test + np.random.normal(0, 1000000, n_samples)

# Преобразуем в DataFrame для удобства
train_data = pd.DataFrame({'area': area_train, 'rooms': rooms_train, 'age': age_train, 'price': price_train})
test_data = pd.DataFrame({'area': area_test, 'rooms': rooms_test, 'age': age_test, 'price': price_test})
test_data_noisy = pd.DataFrame({'area': area_test, 'rooms': rooms_test, 'age': age_test, 'price': price_test_noisy})

# Обучаем модель на нашем тренировочном датасете
model = LinearRegression()
model.fit(train_data[['area', 'rooms', 'age']], train_data['price'])
print(train_data.describe())

#Функция для вычисления R-квадрата
def r2_calc(model, data):
    x_test = data[['area', 'rooms', 'age']]
    y_test = data['price']
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return r2

#Функция для вычисления MAE (Средняя абсолютная ошибка)
def mae_calc(model, data):
    x_test = data[['area', 'rooms', 'age']]
    y_test = data['price']
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


r2_train = r2_calc(model, train_data)
r2_test = r2_calc(model, test_data)
r2_test_noisy = r2_calc(model, test_data_noisy)
mae_train = mae_calc(model, train_data)
mae_test = mae_calc(model, test_data)
mae_test_noisy = mae_calc(model, test_data_noisy)


def test_r2():
    assert ((r2_train > 0.99) & (r2_test > 0.95) & (r2_test_noisy > 0.5))


def test_mae():
    assert ((mae_train < 10000) & (mae_test < 10000) & (mae_test_noisy < 1000000))


def test_r2_comparison():
    assert r2_test_noisy < r2_test < r2_train


def test_mae_comparison():
    assert mae_train < mae_test < mae_test_noisy


print(f'R2 на тренировочном датасете: {r2_train}')
print(f'R2 на тестовом датасете: {r2_test}')
print(f'R2 на датасете с шумом: {r2_test_noisy}')
print(f'Mae на тренировочном датасете: {mae_train}')
print(f'Mae на тестовом датасете: {mae_test}')
print(f'Mae на датасете с шумом: {mae_test_noisy}')
