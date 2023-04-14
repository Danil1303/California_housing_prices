import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def main():
    housing = pd.read_csv('housing.csv')
    # view_info(housing)

    # Распределение на дискретные категории дохода
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    # Объединение полученных категорий с 5 по 10 в одну
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

    # Разбиение на обучающую и тестовую выборки в долях, соответствующих долям вышесозданных категорий в общей выборке
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    # alpha - оставляет выделенными места с высокой плотностью, остальные "размывает"
    # housing.plot(kind='scatter', x='longitude', y='latitude', s=housing['population'] / 100, alpha=0.4,
    #              xlabel='Долгота', ylabel='Широта', label='Население', c='median_house_value',
    #              colormap=plt.get_cmap('jet'), colorbar=True)
    # plt.show()

    # Коэффициент корреляции (Пирсона), показывающий зависимость цены от других признаков
    corr_matrix = housing.drop('ocean_proximity', axis=1, inplace=False).corr()
    # print(corr_matrix['median_house_value'].sort_values(ascending=False))
    # attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    # scatter_matrix(housing[attributes], figsize=(12, 8))
    # housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    # plt.show()

    # Рассчитаем количество комнат на дом
    housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    # Спальни на комнаты
    housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
    # Население на дом
    housing['population_per_household'] = housing['population'] / housing['households']
    corr_matrix = housing.drop('ocean_proximity', axis=1, inplace=False).corr()
    # print(corr_matrix['median_house_value'].sort_values(ascending=False))

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

    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']
    num_pipeline = Pipeline(
        [('selector', DataFrameSelector(num_attribs)),
         ('imputer', SimpleImputer(strategy='median')),
         ('attribs_adder', CombinedAttributesAdder()),
         ('std_scaler', StandardScaler())])
    cat_pipeline = Pipeline(
        [('selector', DataFrameSelector(cat_attribs)),
         ('cat_encoder', OneHotEncoder())])
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)])

    housing_prepared = full_pipeline.fit_transform(housing)

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_reg_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_reg_rmse = np.sqrt(lin_reg_mse)
    print(f'LinearRegression RMSE: {lin_reg_rmse}')
    scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
    lin_reg_scores_rmse = np.sqrt(-scores)
    display_scores(lin_reg_scores_rmse)
    # some_data = housing.iloc[: 5]
    # some_labels = housing_labels.iloc[: 5]
    # some_data_prepared = full_pipeline.transform(some_data)
    # print('Прогнозы линейной регрессии: ', lin_reg.predict(some_data_prepared))
    # print('Метки линейной регрессии: ', list(some_labels))

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_reg_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_reg_rmse = np.sqrt(tree_reg_mse)
    print(f'DecisionTreeRegressor RMSE: {tree_reg_rmse}')
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
    tree_reg_scores_rmse = np.sqrt(-scores)
    display_scores(tree_reg_scores_rmse)

    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_reg_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_reg_rmse = np.sqrt(forest_reg_mse)
    print(f'RandomForestRegressor RMSE: {forest_reg_rmse}')
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
    forest_reg_scores_rmse = np.sqrt(-scores)
    display_scores(forest_reg_scores_rmse)


def view_info(housing):
    housing.info()
    print('\n', housing.describe())
    housing.hist(bins=50, figsize=(20, 9))
    plt.show()


def display_scores(scores):
    print('Суммы оценок : ', scores)
    print('Cpeднee : ', scores.mean())
    print('Стандартное отклонение : ', scores.std())


if __name__ == '__main__':
    main()
