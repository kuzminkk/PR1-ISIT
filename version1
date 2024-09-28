import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

data = pd.read_csv('/content/encoded-data.csv')

def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

data['wake_time'] = data['wake_time'].apply(time_to_minutes)

print(data.head())

label_encoders = {}
categorical_columns = ['gender', 'smoking', 'eye_color', 'well_slept',
                       'chronotype', 'near_coffee_shop', 'gourmet',
                       'office_worker', 'homebody', 'chronic_diseases',
                       'handedness', 'zodiac']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

le_drink = LabelEncoder()
data['preferred_drink'] = le_drink.fit_transform(data['preferred_drink'])


X = data.drop('preferred_drink', axis=1)
y = data['preferred_drink']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Здесь вычисляются среднее и стандартное отклонение для каждого признака на обучающей выборке, и затем все признаки стандартизируются.
X_test_scaled = scaler.transform(X_test) # Для тестовых данных применяется те же параметры стандартизации (среднее и стандартное отклонение), что были рассчитаны на обучающих данных. Это необходимо для того, чтобы сохранить консистентность преобразований.

# Выбор оптимального k с помощью кросс-валидации
k_values = range(1, 21)  # Тестируем значения k от 1 до 20
cross_val_scores = [] # Создаем пусто список для хранения результатов кросс-валидации для каждого значения k

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cross_val_scores.append(scores.mean()) # Вычисляется среднее значение точности для каждого значения k (scores.mean()) и добавляется в список cross_val_scores.

# Найдем лучшее значение k
best_k = k_values[np.argmax(cross_val_scores)] # np.argmax(cross_val_scores) находит индекс максимального значения в списке cross_val_scores, то есть то значение k, для которого средняя точность была максимальной.
print(f'Лучшее значение k: {best_k}')


# Обучение модели k-NN с выбранным значением k
knn = KNeighborsClassifier(n_neighbors=best_k) #Создаётся объект модели k-NN (k-ближайших соседей) из библиотеки scikit-learn, который указывает количество ближайших соседий (k=5)
knn.fit(X_train_scaled, y_train) # Модель обучается на тренировочных данных

# Предсказание
y_pred = knn.predict(X_test_scaled) # Выполняется предсказание для тестовых данных. Результатом является массив y_pred, содержащий предсказанные классы для каждого примера из тестовой выборки




# Пример предсказания для новых данных
# Оценка модели
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

new_data = {
    'gender': 'Мужчина',
    'age': 30,
    'healthy_lifestyle': 80,
    'smoking': 'Нет',
    'eye_color': 'Карий',
    'stress_level': 40,
    'well_slept': 'Да',
    'chronotype': 'Жаворонок',
    'wake_time': '07:00',  # Это нужно будет изменить перед вводом
    'sleep_on_average': 7,
    'near_coffee_shop': 'Да',
    'gourmet': 'Нет',
    'office_worker': 'Да',
    'homebody': 'Нет',
    'chronic_diseases': 'Нет',
    'handedness': 'Правой',
    'zodiac': 'Овен'
}

# Преобразуем новые данные в формат для модели
new_data_df = pd.DataFrame([new_data])
new_data_df['wake_time'] = new_data_df['wake_time'].apply(time_to_minutes)

#Преобразуем категориальные данные
for column in categorical_columns:
    new_data_df[column] = label_encoders[column].transform(new_data_df[column])

# Стандартизируем новые данные
new_data_scaled = scaler.transform(new_data_df)

# Предсказание для новых данных
#Используем метод .predict() для предсказания класса на основе новых данных. Модель возвращает числовое значение, которое соответствует предсказанному классу напитка.
prediction = knn.predict(new_data_scaled)

# Обратное преобразование предсказания в исходную категорию
predicted_drink = le_drink.inverse_transform(prediction) #Получаем имя напитка "Чай" или "Кофе"

print(f'Предсказание для новых данных: {predicted_drink[0]}') #Выводим предсказанный напиток в удобочитаемом виде в консоль
