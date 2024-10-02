# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('2024-09-23 Sotsiologicheskii opros.csv', encoding='cp1251', sep=';')

# Преобразование категориальных признаков
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('Что вы предпочитаете?', axis=1)
y = data['Что вы предпочитаете?']

# Нормализация числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=20)

# Подбор оптимального k
param_grid = {'n_neighbors': list(range(1, 21))}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print(f"Лучшее значение k: {best_k}")

# Обучение модели с оптимальным k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

y_pred_best = knn_best.predict(X_test)

# Оценка точности для лучшего k
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Точность модели с лучшим k ({best_k}): {accuracy_best:.2f}")

# Обучение модели с k=4
knn_k4 = KNeighborsClassifier(n_neighbors=5)
knn_k4.fit(X_train, y_train)

y_pred_k4 = knn_k4.predict(X_test)

# Оценка точности для k=4
accuracy_k4 = accuracy_score(y_test, y_pred_k4)
print(f"Точность модели с k=4: {accuracy_k4:.2f}")
