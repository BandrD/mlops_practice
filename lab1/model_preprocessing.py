import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка обучающих данных
train_data = pd.read_csv("train/build_price_train.csv")

# Выделение признаков и целевой переменной

X_train = train_data.drop(columns=['price_doc'])
y_train = train_data['price_doc']

# Применение предобработки данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Сохранение предобработанных данных
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('train/build_price_train_preprocessed.csv', index=False)
