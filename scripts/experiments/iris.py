import os

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

import spn.structure.leaves.parametric.Parametric as spn_parameter_types
import spn.structure.StatisticalTypes as spn_statistical_types

iris_forward_map = {'species': {0: 'setosa', 1: 'versicolor', 2: 'virginica'}}

iris_backward_map = {'species': {'setosa': 0, 'versicolor': 1, 'virginica': 2}}

feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

spn_parameters = {
    'sepal length': spn_parameter_types.Gaussian,
    'sepal width': spn_parameter_types.Gaussian,
    'petal length':spn_parameter_types.Gaussian,
    'petal width': spn_parameter_types.Gaussian,
    'species': spn_parameter_types.Categorical
}

spn_metatypes = {
        'sepal length': spn_statistical_types.MetaType.REAL,
        'sepal width': spn_statistical_types.MetaType.REAL,
        'petal length':  spn_statistical_types.MetaType.REAL,
        'petal width': spn_statistical_types.MetaType.REAL,
        'species': spn_statistical_types.MetaType.DISCRETE,
}

iris_df = None

def load_dataset():
    global iris_df
    if iris_df is None:
        data = load_iris()
        X = data.data
        y = data.target
        iris_data = np.vstack([X.T, y])
        iris_df = pd.DataFrame(dict(zip(feature_names, iris_data)))
    return iris_df

def iris(continuous=False, discrete_species=True):
    df = load_dataset().copy()
    if continuous:
        del df["species"]
    if not discrete_species:
        df["species"] = pd.Series(df["species"]).map(iris_forward_map["species"])
    return df


def train():
    pass

 #df[feature] = pd.Series(df[feature]).map(feature_map)



if __name__ == '__main__':
    print(iris())
    print(iris(continuous = True))
    print(iris(discrete_species=False))