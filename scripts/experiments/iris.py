import os

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

import spn.structure.leaves.parametric.Parametric as spn_parameter_types
import spn.structure.StatisticalTypes as spn_statistical_types

iris_forward_map = {'species': {0: 'setosa', 1: 'versicolor', 2: 'virginica'}}

iris_backward_map = {'species': {'setosa': 0, 'versicolor': 1, 'virginica': 2}}

feature_names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species']

_numeric_data = os.path.splitext(__file__)[0] + "_numeric.csv"

spn_parameters = {
    'sepallength': spn_parameter_types.Gaussian,
    'sepalwidth': spn_parameter_types.Gaussian,
    'petallength':spn_parameter_types.Gaussian,
    'petalwidth': spn_parameter_types.Gaussian,
    'species': spn_parameter_types.Categorical
}

spn_metatypes = {
        'sepallength': spn_statistical_types.MetaType.REAL,
        'sepalwidth': spn_statistical_types.MetaType.REAL,
        'petallength':  spn_statistical_types.MetaType.REAL,
        'petalwidth': spn_statistical_types.MetaType.REAL,
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
    df = iris(discrete_species=False)
    print(df)
