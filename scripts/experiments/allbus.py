import pandas as pd
import os

import spn.structure.leaves.parametric.Parametric as spn_parameter_types
import spn.structure.StatisticalTypes as spn_statistical_types

_test_data = os.path.splitext(__file__)[0] + "_test.csv"
_train_data = os.path.splitext(__file__)[0] + "_train.csv"
_numeric_data = os.path.splitext(__file__)[0] + "_train_numeric.csv"

allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                      'lived_abroad': {'No': 0, 'Yes': 1},
                      'happiness': {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8,
                                    'h9': 9, 'h10': 10}}

allbus_backward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                       'lived_abroad': {0: 'No', 1: 'Yes'},
                       'happiness': {0: 'h0', 1: 'h1', 2: 'h2', 3: 'h3', 4: 'h4', 5: 'h5', 6: 'h6', 7: 'h7', 8: 'h8',
                                     9: 'h9', 10: 'h10'}}

spn_parameters = {
    'julien': {
        'sex': spn_parameter_types.Categorical,
        'eastwest': spn_parameter_types.Categorical,
        'lived_abroad': spn_parameter_types.Categorical,
        'age': spn_parameter_types.Gaussian,
        'educ': spn_parameter_types.Gaussian,
        'income': spn_parameter_types.Gaussian,
        'happiness': spn_parameter_types.Gaussian,
        'health': spn_parameter_types.Gaussian
    }
}

spn_metatypes = {
    'julien': {
        'sex': spn_statistical_types.MetaType.DISCRETE,
        'eastwest': spn_statistical_types.MetaType.DISCRETE,
        'lived_abroad': spn_statistical_types.MetaType.DISCRETE,
        'age': spn_statistical_types.MetaType.REAL,
        'educ': spn_statistical_types.MetaType.DISCRETE,
        'income': spn_statistical_types.MetaType.REAL,
        'happiness': spn_statistical_types.MetaType.REAL,
        'health': spn_statistical_types.MetaType.REAL
    }
}

def train(filepath=_train_data, discrete_happy=True, discretize_all=False):
    df = pd.read_csv(filepath)
    if discretize_all:
        for feature, feature_map in allbus_forward_map.items():
            df[feature] = pd.Series(df[feature]).map(feature_map)
        discrete_happy = True
    if not discrete_happy:
        happiness_map = {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8, 'h9': 9,
                         'h10': 10}
        df["happiness"] = pd.Series(df["happiness"]).map(happiness_map)
    return df


def test(filepath=_test_data, discrete_happy=True, discretize_all=False):
    df = pd.read_csv(filepath)
    if discretize_all:
        for feature, feature_map in allbus_forward_map.items():
            df[feature] = pd.Series(df[feature]).map(feature_map)
        discrete_happy = True
    if not discrete_happy:
        happiness_map = {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8, 'h9': 9,
                         'h10': 10}
        df["happiness"] = pd.Series(df["happiness"]).map(happiness_map)
    return df


def train_reduced(filepath=_train_data, discrete_happy=True):
    df = pd.read_csv(filepath).drop(["lived_abroad", "health"], axis=1)
    if not discrete_happy:
        happiness_map = {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8, 'h9': 9,
                         'h10': 10}
        df["happiness"] = pd.Series(df["happiness"]).map(happiness_map)
    return df


def test_reduced(filepath=_test_data, discrete_happy=True):
    df = pd.read_csv(filepath).drop(["lived_abroad", "health"], axis=1)
    if not discrete_happy:
        happiness_map = {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8, 'h9': 9,
                         'h10': 10}
        df["happiness"] = pd.Series(df["happiness"]).map(happiness_map)
    return df


def train_continuous(filepath=_train_data):
    df = pd.read_csv(filepath).drop(['sex', 'eastwest', 'happiness', 'lived_abroad'], axis=1)
    return df


def test_continuous(filepath=_test_data):
    df = pd.read_csv(filepath).drop(['sex', 'eastwest', 'happiness', 'lived_abroad'], axis=1)
    return df


if __name__ == '__main__':
    allbus_file_all_discrete = train(discretize_all=True)
    allbus_file_all_discrete.to_csv(
        "/home/julien/PycharmProjects/modelbase/scripts/experiments/allbus_train_numeric.csv", index=False)
