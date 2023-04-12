import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


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
    housing.plot(kind='scatter', x='longitude', y='latitude', s=housing['population'] / 100, alpha=0.4,
                 xlabel='Долгота', ylabel='Широта', label='Население', c='median_house_value',
                 colormap=plt.get_cmap('jet'), colorbar=True)
    plt.show()

    # Коэффициент корреляции (Пирсона), показывающий зависимость цены от других признаков
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)


def view_info(housing):
    housing.info()
    print('\n', housing.describe())
    housing.hist(bins=50, figsize=(20, 9))
    plt.show()


if __name__ == '__main__':
    main()
