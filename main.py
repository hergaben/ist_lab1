import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Подготовка данных
data = pd.read_csv('датасет_кофе_чай.csv', encoding='cp1251', sep=';')

# Преобразование времени пробуждения в минуты от начала дня
data['wake_up'] = data['wake_up'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

# Определение признаков и целевой переменной
X = data[['wake_up', 'weather', 'feeling', 'mood', 'availability_coffee', 'availability_tea', 'day_of_the_week']].copy()
y = data['drink']

# Преобразование категориальных признаков в бинарные (one-hot)
X = pd.get_dummies(X, columns=['weather', 'day_of_the_week', 'availability_coffee', 'availability_tea', 'mood'], drop_first=True)

# Преобразование целевой переменной
y = y.map({'чай': 0, 'кофе': 1})

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание и обучение модели KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Прогнозирование
y_pred = knn.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

# Пример прогнозирования для новых данных
# Необходимо создать DataFrame с теми же признаками, что и обучающая выборка
new_data_raw = {
    'wake_up': '7:30',  # Время пробуждения
    'weather': 'солнечно',
    'feeling': 8,
    'mood': 'отличное',
    'availability_coffee': 'да',
    'availability_tea': 'нет',
    'day_of_the_week': 'среда'
}

# Преобразование времени
wake_up_minutes = int(new_data_raw['wake_up'].split(':')[0]) * 60 + int(new_data_raw['wake_up'].split(':')[1])

# Создание DataFrame
new_data = pd.DataFrame({
    'wake_up': [wake_up_minutes],
    'weather': [new_data_raw['weather']],
    'feeling': [new_data_raw['feeling']],
    'mood': [new_data_raw['mood']],
    'availability_coffee': [new_data_raw['availability_coffee']],
    'availability_tea': [new_data_raw['availability_tea']],
    'day_of_the_week': [new_data_raw['day_of_the_week']]
})

# Преобразование категориальных признаков с помощью get_dummies
new_data = pd.get_dummies(new_data, columns=['weather', 'day_of_the_week', 'availability_coffee', 'availability_tea', 'mood'], drop_first=True)

# Обеспечение наличия всех столбцов, как в обучающей выборке
# Если какого-то столбца нет в new_data, добавить его со значением 0
for col in X.columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Упорядочение столбцов
new_data = new_data[X.columns]

# Масштабирование новых данных
new_data_scaled = scaler.transform(new_data)

# Прогнозирование
prediction = knn.predict(new_data_scaled)
predicted_drink = 'кофе' if prediction[0] == 1 else 'чай'
print(f'Прогнозируемый напиток: {predicted_drink}')

k_values = range(1, 20)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[cv_scores.index(max(cv_scores))]
print(f'Оптимальное значение k: {optimal_k}')
