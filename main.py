import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_features = lambda data: [column for column in data.columns if data[column].dtype in [int, float]]

melbourne_file_path = 'melb_data.csv'
data = pd.read_csv(melbourne_file_path)


scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
                     greater_is_better=False)


# Посмотрим на результат без предварительной работы с данными, используя только числовые столбцы, не подбирая параметров для модели

model = RandomForestRegressor()
y = data['Price']
X = data[num_features(data)].drop(['Price'], axis=1)

scores = -1 * cross_val_score(model, X, y,
                              cv=5,
                              scoring=scorer)
print("Average RMSE score:", scores.mean())


for ax, feature in zip(plt.subplots(4, 3, figsize=(20,20))[1].flatten(), num_features(data)):
    ax.set_title(feature)
    ax.boxplot(data[feature])


# Смотрим где и сколько пропущенных значений + какого они типа
print(data.isna().sum().to_frame(name='Количество пропущенных').join(data.dtypes.to_frame(name='Тип данных')))

# Удаление пропущенных значений
# В BuildingArea почти половина значений пропущены - будет логично удалить этот столбец
# В YearBuilt так же много пропущенных значений - удаляем
data = melbourne_data.drop(['BuildingArea', 'YearBuilt'], axis=1)


# Очистим Данные от выбросов
data.drop(data[data['Rooms'] > 4].index, inplace=True)
data.drop(data[(data['Landsize'] == 0) | (data['Landsize'] >= 1000)].index, inplace=True)
data['Car'].fillna(0, inplace=True)
data.drop(data[data['Car'] > 3].index, inplace=True)
data.drop(data[data['Price'] > 2000000].index, inplace=True)
data.drop(data[data['Distance'] > 22].index, inplace=True)
data.drop(data[data['Postcode'] > 3200].index, inplace=True)
data.drop(data[(data['Bedroom2'] > 4) | (data['Bedroom2'] == 0)].index, inplace=True)
data.drop(data[data['Bathroom'] > 3].index, inplace=True)
data.drop(data[(data['Lattitude'] > -37.6) | (data['Lattitude'] <= -38)].index, inplace=True)
data.drop(data[data['Longtitude'] < 144.7].index, inplace=True)
data.drop(data[data['Propertycount'] > 16000].index, inplace=True)

for ax, feature in zip(plt.subplots(4, 3, figsize=(20,20))[1].flatten(), num_features(data)):
    ax.set_title(feature)
    ax.boxplot(data[feature])


y = data['Price']
X = data[num_features(data)].drop(['Price'], axis=1)

scores = -1 * cross_val_score(pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print("Average RMSE score (across experiments):", scores.mean())


y = data['Price']
X = data[num_features(data)].drop(['Price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 4)


#Воспользуемся сеткой, чтобы подобрать лучшие параметры

# Модель
model = RandomForestRegressor(random_state=42)

# Сетка параметров
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

# Поиск по сетке
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Лучшие параметры
print(grid_search.best_params_)


# In[438]:


#Воспользуемся сеткой, чтобы подобрать лучшие параметры

# Модель
model = RandomForestRegressor(random_state=42)

# Сетка параметров
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

# Поиск по сетке
grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring=scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Лучшие параметры
print(grid_search.best_params_)


model = RandomForestRegressor(**grid_search.best_params_)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
MAE = mean_absolute_error(y_train, train_preds)
MSE = mean_squared_error(y_train, train_preds)
RMSE = np.sqrt(MSE)
print(f'Train:\n\tmean absolute error = {MAE};\n\troot mean squared error = {RMSE};\n\tmedian absolute error = {MedianAE}')

test_preds = forest_model.predict(test_X)
MAE = mean_absolute_error(test_y, test_preds)
MSE = mean_squared_error(test_y, test_preds)
RMSE = np.sqrt(MSE)
MedianAE = median_absolute_error(test_y, test_preds)
print(f'Test:\n\tmean absolute error = {MAE};\n\troot mean squared error = {RMSE};\n\tmedian absolute error = {MedianAE}')
