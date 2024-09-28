import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

#Подготовка данных
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
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

# Выбор оптимального k с помощью кросс-валидации
k_values = range(1, 21)  
cross_val_scores = [] 

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cross_val_scores.append(scores.mean()) 

best_k = k_values[np.argmax(cross_val_scores)] 
print(f'Лучшее значение k: {best_k}')


# Обучение модели k-NN с выбранным значением k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train) 

y_pred = knn.predict(X_test_scaled) 

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

#Тестирование
new_data = {
    'gender': 'Мужчина',
    'age': 30,
    'healthy_lifestyle': 80,
    'smoking': 'Нет',
    'eye_color': 'Карий',
    'stress_level': 40,
    'well_slept': 'Да',
    'chronotype': 'Жаворонок',
    'wake_time': '07:00', 
    'sleep_on_average': 7,
    'near_coffee_shop': 'Да',
    'gourmet': 'Нет',
    'office_worker': 'Да',
    'homebody': 'Нет',
    'chronic_diseases': 'Нет',
    'handedness': 'Правой',
    'zodiac': 'Овен'
}

new_data_df = pd.DataFrame([new_data])
new_data_df['wake_time'] = new_data_df['wake_time'].apply(time_to_minutes)

for column in categorical_columns:
    new_data_df[column] = label_encoders[column].transform(new_data_df[column])

new_data_scaled = scaler.transform(new_data_df)

prediction = knn.predict(new_data_scaled)

predicted_drink = le_drink.inverse_transform(prediction) 

print(f'Предсказание для новых данных: {predicted_drink[0]}') 
