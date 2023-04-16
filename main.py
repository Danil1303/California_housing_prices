import joblib
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import randint
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV


def main() -> None:
    housing = pd.read_csv('housing.csv')
    # Ограничение размера выборки в целях ускорения расчётов
    housing = housing[:8315]

    view_info(housing)

    # Распределение на дискретные категории дохода
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    # Объединение полученных категорий с 5 по 10 в одну
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

    # Разбиение на обучающую и тестовую выборки в долях, соответствующим долям для вышесозданного атрибута в общей выборке
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    # alpha - оставляет выделенными места с высокой плотностью, остальные "размывает"
    housing.plot(kind='scatter', x='longitude', y='latitude', s=housing['population'] / 100, alpha=0.4,
                 xlabel='Долгота', ylabel='Широта', label='Население', c='median_house_value',
                 colormap=plt.get_cmap('jet'), colorbar=True)
    plt.show()

    # Коэффициент корреляции (Пирсона), показывающий зависимость цены от других атрибутов
    corr_matrix = housing.drop('ocean_proximity', axis=1, inplace=False).corr()
    print('\nКоэффициенты корреляции цены от других атрибутов:\n',
          corr_matrix['median_house_value'].sort_values(ascending=False))
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(housing[attributes], figsize=(12, 8))
    housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    plt.show()

    # Рассчитаем количество комнат на дом
    housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    # Спальни на комнаты
    housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
    # Население на дом
    housing['population_per_household'] = housing['population'] / housing['households']
    corr_matrix = housing.drop('ocean_proximity', axis=1, inplace=False).corr()
    print('\nКоэффициенты корреляции цены от других атрибутов с учётом синтезированных атрибутов:\n',
          corr_matrix['median_house_value'].sort_values(ascending=False))

    housing = strat_train_set.drop('median_house_value', axis=1)
    housing_labels = strat_train_set['median_house_value']

    # Заполнение недостающих данных в столбце total_bedrooms средними значениями
    '''
    median = housing['total_bedrooms'].median()
    housing['total_bedrooms'].fillna(median, inplace=True)
    '''
    # Или так:
    imputer = SimpleImputer(strategy='median')
    housing_num = housing.drop('ocean_proximity', axis=1)
    # Экземпляр imputer подсчитывает медиану каждого атрибута и
    # сохраняет результат в своей переменной экземпляра statistics:
    imputer.fit(housing_num)
    # transform дополняет отсутствующие значения в housing_num средними из imputer:
    values = imputer.transform(housing_num)
    # На выходе получается numpy массив, который нужно запихнуть в датафрейм:
    housing_tr = pd.DataFrame(values, columns=housing_num.columns)

    # Присвоение категориальному атрибуту целочисленных значений
    housing_cat = housing['ocean_proximity']
    housing_cat_encoded, housing_categories = housing_cat.factorize()
    # Использование кодирования с одним активным состоянием
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(- 1, 1))

    # Получение индексов столбцов
    rooms_ix, bedrooms_ix, population_ix, household_ix = [
        list(housing.columns).index(col)
        for col in ('total_rooms', 'total_bedrooms', 'population', 'households')]

    def add_extra_features(X: np.ndarray) -> np.ndarray:
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

    # Конвейер трансформации данных
    housing_num = housing.drop('ocean_proximity', axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False, )),
        ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([
        ('num_attribs', num_pipeline, num_attribs),
        ('cat_attribs', OneHotEncoder(), cat_attribs)])
    housing_prepared = full_pipeline.fit_transform(housing)

    # Линейная регрессия
    if os.path.isfile('pickle_models/lin_reg_model.pkl'):
        lin_reg = joblib.load('pickle_models/lin_reg_model.pkl')
    else:
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        joblib.dump(lin_reg, 'pickle_models/lin_reg_model.pkl')
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_reg_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_reg_rmse = np.sqrt(lin_reg_mse)
    print('\nLinearRegression RMSE: ', lin_reg_rmse)
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
    lin_reg_scores_rmse = np.sqrt(-scores)
    display_scores(lin_reg_scores_rmse)
    some_data = housing.iloc[: 5]
    some_labels = housing_labels.iloc[: 5]
    some_data_prepared = full_pipeline.transform(some_data)
    print('Прогнозы линейной регрессии: ', lin_reg.predict(some_data_prepared))
    print('Метки линейной регрессии: ', list(some_labels))

    # Дерево решений для регрессии
    if os.path.isfile('pickle_models/tree_reg_model.pkl'):
        tree_reg = joblib.load('pickle_models/tree_reg_model.pkl')
    else:
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(housing_prepared, housing_labels)
        joblib.dump(tree_reg, 'pickle_models/tree_reg_model.pkl')
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_reg_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_reg_rmse = np.sqrt(tree_reg_mse)
    print('\nDecisionTreeRegressor RMSE: ', tree_reg_rmse)
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
    tree_reg_scores_rmse = np.sqrt(-scores)
    display_scores(tree_reg_scores_rmse)

    # Случайный лес для регрессии
    if os.path.isfile('pickle_models/forest_reg_model.pkl'):
        forest_reg = joblib.load('pickle_models/forest_reg_model.pkl')
    else:
        forest_reg = RandomForestRegressor(random_state=42)
        forest_reg.fit(housing_prepared, housing_labels)
        joblib.dump(forest_reg, 'pickle_models/forest_reg_model.pkl')
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_reg_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_reg_rmse = np.sqrt(forest_reg_mse)
    print('\nRandomForestRegressor RMSE: ', forest_reg_rmse)
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
    forest_reg_scores_rmse = np.sqrt(-scores)
    display_scores(forest_reg_scores_rmse)

    # Решётчатый поиск наилучших гиперпараметров случайного леса
    param_grid = [{'n_estimators': [3, 10, 20], 'max_features': [2, 4, 6, 8]},
                  {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)
    housing_predictions = grid_search.predict(housing_prepared)
    grid_search_mse = mean_squared_error(housing_labels, housing_predictions)
    grid_search_rmse = np.sqrt(grid_search_mse)
    print('\nRandomForestRegressor решётчатый поиск RMSE: ', grid_search_rmse)
    print('Лучшие гиперпараметры для DecisionTreeRegressor, найденные при помощи решётчатого поиска:',
          grid_search.best_params_)
    # Вывод влияния атрибутов по результатам решётчатого поиска для случайного леса
    feature_importances = grid_search.best_estimator_.feature_importances_
    extra_attribs = ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
    cat_encoder = full_pipeline.named_transformers_['cat_attribs']
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print('Влияние каждого атрибута для решётчатого поиска:')
    for attr in sorted(zip(feature_importances, attributes), reverse=True):
        print(attr)

    # Случайный поиск наилучших гиперпараметров случайного леса
    random_search_param = {'n_estimators': randint(low=1, high=50), 'max_features': randint(low=1, high=8)}
    forest_reg = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(forest_reg, param_distributions=random_search_param,
                                       n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(housing_prepared, housing_labels)
    housing_predictions = random_search.predict(housing_prepared)
    random_search_mse = mean_squared_error(housing_labels, housing_predictions)
    random_search_rmse = np.sqrt(random_search_mse)
    print('\nRandomForestRegressor случайный поиск RMSE: ', random_search_rmse)
    print('Лучшие гиперпараметры для DecisionTreeRegressor, найденные при помощи случайного поиска:',
          random_search.best_params_)
    # Вывод влияния атрибутов по результатам случайного поиска для случайного леса
    feature_importances = random_search.best_estimator_.feature_importances_
    extra_attribs = ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
    cat_encoder = full_pipeline.named_transformers_['cat_attribs']
    cat_one_hot_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print('Влияние каждого атрибута для случайного поиска:')
    for attr in sorted(zip(feature_importances, attributes), reverse=True):
        print(attr)

    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop('median_house_value', axis=1)
    y_test = strat_test_set['median_house_value'].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('\nRMSE финальной модели: ', final_rmse)

    # Расчёт 95% доверительного интервала для RMSE
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    error = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                     loc=squared_errors.mean(),
                                     scale=stats.sem(squared_errors)))
    print('95% доверительный интервал для RMSE: ', error)

    # Расчёт интервала другим способом
    m = len(squared_errors)
    mean = squared_errors.mean()
    tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
    tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
    np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

    zscore = stats.norm.ppf((1 + confidence) / 2)
    zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
    np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

    # z-метрика вместо t-метрики
    zscore = stats.norm.ppf((1 + confidence) / 2)
    zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
    np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

    # Регрессия при помощи метода опорных векторов
    param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0], 'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}]
    svm_reg = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    svm_reg.fit(housing_prepared, housing_labels)
    housing_predictions = svm_reg.predict(housing_prepared)
    svm_reg_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_reg_rmse = np.sqrt(svm_reg_mse)
    print('\nSVR RMSE: ', svm_reg_rmse)

    negative_mse = svm_reg.best_score_
    rmse = np.sqrt(-negative_mse)
    print('SVR RMSE v2: ', rmse)


def view_info(housing: pd.DataFrame) -> None:
    housing.info()
    print('\n', housing.describe())
    housing.hist(bins=50, figsize=(20, 9))
    plt.show()


def display_scores(scores: np.ndarray) -> None:
    print('Суммы оценок : ', scores)
    print('Cpeднee : ', scores.mean())
    print('Стандартное отклонение : ', scores.std(), '\n')


if __name__ == '__main__':
    main()
