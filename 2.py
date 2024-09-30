# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter


# Функция для расчета Евклидова расстояния
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Реализация K-ближайших соседей
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Вычисление расстояний от x до всех точек обучающей выборки
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Сортировка и выбор K ближайших соседей
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Возвращаем наиболее частую метку
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Шаг 1: Загрузка данных
file_path = '2024-09-23 Sotsiologicheskii opros.csv'
data = pd.read_csv(file_path, encoding='cp1251', sep=';')

# Шаг 2: Предварительная обработка данных
# Заполнение пропущенных значений медианой
data.fillna(data.median(numeric_only=True), inplace=True)

# Преобразуем категориальные признаки в числовые
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.factorize(data[column])[0]

# Шаг 3: Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Что вы предпочитаете?'])  # Признаки
y = data['Что вы предпочитаете?']  # Целевая переменная

# Нормализация данных
X = (X - X.mean()) / X.std()


# Шаг 5: Разделение данных на обучающую и тестовую выборки
def train_test_split(X, y, test_size=0.2):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size = int(len(X) * test_size)
    X_train, X_test = X.iloc[indices[:-test_size]], X.iloc[indices[-test_size:]]
    y_train, y_test = y.iloc[indices[:-test_size]], y.iloc[indices[-test_size:]]
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X, y)

# Шаг 6: Обучение модели
knn = KNN(k=5)
knn.fit(np.array(X_train), np.array(y_train))

# Шаг 7: Прогнозирование
y_pred = knn.predict(np.array(X_test))

# Оценка точности
accuracy = np.sum(y_pred == np.array(y_test)) / len(y_test)
print(f'Точность модели: {accuracy * 100:.2f}%')


# Шаг 8: Прогнозирование для новых данных
def predict_drink(new_data):
    # Обработка новых данных (заполнение пропусков и нормализация)
    new_data.fillna(new_data.median(numeric_only=True), inplace=True)

    for column in new_data.columns:
        if new_data[column].dtype == 'object':
            new_data[column] = pd.factorize(new_data[column])[0]

    new_data_normalized = (new_data - X.mean()) / X.std()

    # Прогнозирование
    return knn.predict(np.array(new_data_normalized))

