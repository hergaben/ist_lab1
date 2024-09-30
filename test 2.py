# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
from collections import Counter

# Ваши функции и классы здесь

class TestKNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Создаем небольшой тестовый датасет
        cls.X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        cls.y_train = np.array([0, 1, 0, 1, 0])

        cls.X_test = np.array([[1.5, 2.5], [4.5, 5.5]])
        cls.y_test_true = np.array([0, 0])  # Ожидаемый результат

        cls.knn = KNN(k=3)
        cls.knn.fit(cls.X_train, cls.y_train)

    def test_euclidean_distance(self):
        # Тестирование функции евклидова расстояния
        x1 = np.array([1, 2])
        x2 = np.array([4, 6])
        result = euclidean_distance(x1, x2)
        expected = 5.0  # Ручной расчет расстояния
        self.assertEqual(result, expected)

    def test_predict(self):
        # Проверка прогноза на тестовых данных
        y_pred = self.knn.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, self.y_test_true)

    def test_accuracy(self):
        # Тестирование точности модели
        y_pred = self.knn.predict(self.X_train)
        accuracy = np.sum(y_pred == self.y_train) / len(self.y_train)
        self.assertGreaterEqual(accuracy, 0.8)  # Ожидаемая точность не менее 80%

    def test_predict_new_data(self):
        # Проверка функции predict_drink
        new_data = pd.DataFrame({
            'Feature1': [3],
            'Feature2': [4]
        })
        result = predict_drink(new_data)
        expected = np.array([0])  # На основе имеющихся данных
        np.testing.assert_array_equal(result, expected)

    def test_data_processing(self):
        # Проверка обработки данных (заполнение пропусков и факторизация)
        data = pd.DataFrame({
            'Feature1': [1, 2, np.nan, 4],
            'Feature2': ['A', 'B', 'A', 'B']
        })

        # Применение таких же шагов, как в основном коде
        data.fillna(data.median(numeric_only=True), inplace=True)
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = pd.factorize(data[column])[0]

        # Проверка, что пропуски заполнены и строки кодируются числами
        self.assertFalse(data.isnull().values.any())
        self.assertTrue(np.issubdtype(data['Feature2'].dtype, np.number))

if __name__ == '__main__':
    unittest.main()
