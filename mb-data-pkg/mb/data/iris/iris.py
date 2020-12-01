# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for the iris data set
"""

import pandas as pd
import os

try:
    import spn.structure.leaves.parametric.Parametric as spn_parameter_types
    import spn.structure.StatisticalTypes as spn_statistical_types
except ImportError:
    pass

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"
_csvfilepath_disc = os.path.splitext(__file__)[0] + "_discretized.csv"


def spn_parameters():
    iris_variable_types = {
        'sepal_length': spn_parameter_types.Gaussian,
        'sepal_width': spn_parameter_types.Gaussian,
        'petal_length': spn_parameter_types.Gaussian,
        'petal_width': spn_parameter_types.Gaussian,
        'species': spn_parameter_types.Categorical
    }
    return iris_variable_types

def spn_metaparameters():
    iris_meta_types = {
        'sepal_length': spn_statistical_types.MetaType.REAL,
        'sepal_width': spn_statistical_types.MetaType.REAL,
        'petal_length': spn_statistical_types.MetaType.REAL,
        'petal_width': spn_statistical_types.MetaType.REAL,
        'species': spn_statistical_types.MetaType.DISCRETE
    }
    return iris_meta_types


def mixed(filepath=_csvfilepath):
    """Loads the iris data set from a csv file, removes the index column and returns the
    remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath)
    return df


def continuous(filepath=_csvfilepath):
    df = mixed(filepath)
    del df['species']
    return df


def discretized(filepath=_csvfilepath_disc):
    df = pd.read_csv(filepath)
    df['species'] = df['species'].astype('category')
    return df


if __name__ == '__main__':
    pass
