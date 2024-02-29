!/bin/bash

# Запуск скрипта для создания данных
python data_creation.py

# Запуск скрипта для предобработки данных
python model_preprocessing.py

# Запуск скрипта для подготовки модели
python model_preparation.py

# Запуск скрипта для тестирования модели
python model_testing.py