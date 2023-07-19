import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загрузка датасета Titanic (в формате CSV)
data = pd.read_csv('titanic.csv')

# Удаление ненужных столбцов
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Заполнение пропущенных значений
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Преобразование категориальных признаков в числовые
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# Разделение данных на признаки и метки классов
X = data.drop('Survived', axis=1)
y = data['Survived']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение решающего дерева
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Прогнозирование на тестовых данных
y_pred = clf.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
