# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Загрузка данных
data = pd.read_csv('датасет_кофе_чай.csv', encoding='cp1251', sep=';')

# Преобразование времени 'wake_up' в количество минут с начала дня
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

data['wake_up'] = data['wake_up'].apply(time_to_minutes)

# Преобразование категориальных признаков
label_encoders = {}
for column in ['weather', 'availability_coffee', 'availability_tea', 'day_of_the_week', 'mood']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Признаки и целевая переменная
X = data.drop('drink', axis=1)
y = data['drink']

# Нормализация числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=55)

# Подбор оптимального k с помощью GridSearchCV
param_grid = {'n_neighbors': list(range(1, 21))}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Лучшая модель
best_k = grid_search.best_params_['n_neighbors']
print(f"Лучшее значение k: {best_k}")

# Обучение модели с оптимальным k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Прогнозирование
y_pred = knn_best.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")
