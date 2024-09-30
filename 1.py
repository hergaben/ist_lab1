import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
file_path = '2024-09-23 Sotsiologicheskii opros.csv'
data = pd.read_csv(file_path, encoding='cp1251', sep=';')

# Предварительная обработка данных
# Преобразуем категориальные признаки в числовые с помощью LabelEncoder
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Что вы предпочитаете?'])  # Признаки
y = data['Что вы предпочитаете?']  # Целевая переменная

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Обучение модели KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Точность модели: {accuracy * 100:.2f}%')


# Прогнозирование на новых данных
def predict_drink(new_data):
    # Преобразуем новые данные так же, как и исходные данные
    for column, encoder in label_encoders.items():
        new_data[column] = encoder.transform(new_data[column])

    # Нормализация новых данных
    new_data_scaled = scaler.transform(new_data)

    # Прогнозирование
    return knn.predict(new_data_scaled)
