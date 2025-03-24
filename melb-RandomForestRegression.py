import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

num_features = lambda data: [column for column in data.columns if data[column].dtype in [int, float]]

melbourne_file_path = 'melb_data.csv'
data = pd.read_csv(melbourne_file_path)


scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
                     greater_is_better=False)



def test_model(mark, data, results='', searchCV=False):
    # Подготовим данные
    X = data.drop('Price', axis=1)  # Все столбцы, кроме 'Price'
    y = data['Price']  # Только столбец 'Price'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    # #Воспользуемся сеткой, чтобы подобрать лучшие параметры
    # # Модель
    model = RandomForestRegressor(random_state=4)
    if searchCV:
        # Сетка параметров
        param_grid = {
            'n_estimators': np.arange(100, 250, 5),
            'max_depth': np.arange(10, 25),
            'min_samples_split': np.arange(2, 5),
            'max_features': [None, 'sqrt', 'log2']
        }
        # Поиск по сетке
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring=scorer, n_jobs=-1)
        grid_search.fit(X_train_standardized, y_train)
        model = RandomForestRegressor(**grid_search.best_params_)

    
    model.fit(X_train_standardized, y_train)

    train_preds = model.predict(X_train_standardized)
    train_MAE = mean_absolute_error(y_train, train_preds)
    train_MSE = mean_squared_error(y_train, train_preds)
    train_RMSE = np.sqrt(train_MSE)
    train_MedianAE = median_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)


    test_preds = model.predict(X_test_standardized)
    test_MAE = mean_absolute_error(y_test, test_preds)
    test_MSE = mean_squared_error(y_test, test_preds)
    test_RMSE = np.sqrt(test_MSE)
    test_MedianAE = median_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    results += f"""
    \t\t{mark} errors:\n
Train:\n\tmean absolute error = {np.round(train_MAE, 2)};\n\troot mean squared error = {np.round(train_RMSE, 2)};\n\tmedian absolute error = {np.round(train_MedianAE, 2)}\n\tr2_score = {np.round(train_r2, 2)}
Test:\n\tmean absolute error = {np.round(test_MAE, 2)};\n\troot mean squared error = {np.round(test_RMSE, 2)};\n\tmedian absolute error = {np.round(test_MedianAE, 2)}\n\tr2_score = {np.round(test_r2, 2)}
"""
    return results

results = test_model('Start', data[num_features(data)])
print(results)


num_features_list = num_features(data)

for column in num_features_list:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = len(data[(data[column] < lower_bound) | (data[column] > upper_bound)])
    print(f'Column {column}; outliers count is {outliers};')


num_data = data[num_features(data)].copy()
num_data.info()
fig, axes = plt.subplots(4, 3, figsize=(20,20))

for ax, feature in zip(axes.flatten(), num_data.columns.to_list()):
    ax.set_title(feature)
    ax.boxplot(data[feature])


num_data = data[num_features(data)].copy()

# Смотрим где и сколько пропущенных значений + какого они типа
print(num_data.isna().sum().to_frame(name='Количество пропущенных').join(num_data.dtypes.to_frame(name='Тип данных')))

# Удаление пропущенных значений
# В BuildingArea почти половина значений пропущены - будет логично удалить этот столбец
# В YearBuilt так же много пропущенных значений - удаляем
num_data.drop(['BuildingArea', 'YearBuilt'], axis=1, inplace=True)

results += test_model('Intermidiate 1 (remove BuildingArea and YearBuilt)', num_data)
print(results)


# Незаполненные значения в Car заполним нулями
num_data['Car'].fillna(0, inplace=True)
results += test_model('Intermidiate 2 (fill nan in Car)', num_data)
print(results)


# Очистка от выбросов
# Очистим Rooms от выбросов
num_data.drop(num_data[num_data['Rooms'] > 4].index, inplace=True)
# В Landsize много выбросов и нулевых значений - уберем их
num_data.drop(num_data[(num_data['Landsize'] == 0) | (num_data['Landsize'] >= 1000)].index, inplace=True)
# Очистим Car от выбросов
num_data.drop(num_data[num_data['Car'] > 3].index, inplace=True)
num_data.drop(num_data[num_data['Price'] > 2000000].index, inplace=True)
num_data.drop(num_data[num_data['Distance'] > 22].index, inplace=True)
num_data.drop(num_data[num_data['Postcode'] > 3200].index, inplace=True)
num_data.drop(num_data[(num_data['Bedroom2'] > 4) | (num_data['Bedroom2'] == 0)].index, inplace=True)
num_data.drop(num_data[num_data['Bathroom'] > 3].index, inplace=True)
num_data.drop(num_data[(num_data['Lattitude'] > -37.6) | (num_data['Lattitude'] <= -38)].index, inplace=True)
num_data.drop(num_data[num_data['Longtitude'] < 144.7].index, inplace=True)
num_data.drop(num_data[num_data['Propertycount'] > 16000].index, inplace=True)

results += test_model('Intermidiate 3 (remove outliers)', num_data)
print(results)


for ax, feature in zip(plt.subplots(4, 3, figsize=(20,20))[1].flatten(), num_data.columns.to_list()):
    ax.set_title(feature)
    ax.boxplot(num_data[feature])


# Для type сделаем OrdinalEncoder - по той логике, что, обычно, дома таунхаузы дешевле дуплесов, а дуплесы дешевле домов/коттеджей
cat_data = pd.DataFrame()
encoder = OrdinalEncoder(categories=[['t', 'u', 'h']])
encoded_data = encoder.fit_transform(data[['Type']])
cat_data['Type'] = encoded_data.ravel()

suburb_frequency = data['Suburb'].value_counts(normalize=True)
cat_data['Suburb_encoded'] = data['Suburb'].map(suburb_frequency)
final_data = num_data.join(cat_data)

results += test_model('Intermidiate 4 (added categorical variables)', final_data)
print(results)


results += test_model('Finale results with grid search', final_data, searchCV=True)
print(results)

