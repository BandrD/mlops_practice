# MLOps_practice

- Падеров Анатолий Викторович РИМ-130907

- Андреев Иван Олегович РИМ-130907

- Бобрешов Виталий Сергеевич РИМ-130906

- Деревянкин Александр Александрович РИМ-130907

---

## lab 1

1.	Создайте python-скрипт (_data_creation.py_), который создает различные наборы данных, описывающие некий процесс (например, изменение дневной температуры). Таких наборов должно быть несколько, в некоторые данные можно включить аномалии или шумы. Можно взять готовый датасет, и выкачать его из интернета. Часть наборов данных должны быть сохранены в папке “train”, другая часть в папке “test”.

2.	создайте python-скрипт (_model_preprocessing.py_), который выполняет предобработку данных, например, с помощью sklearn.preprocessing.StandardScaler.

3.	создайте python-скрипт (_model_preparation.py_), который создает и обучает модель машинного обучения на построенных данных из папки “train”.

4.	создайте python-скрипт (_model_testing.py_), проверяющий модель машинного обучения на построенных данных из папки “test”.

5.	Напишите bash-скрипт (_pipeline.sh_), последовательно запускающий все python-скрипты.
