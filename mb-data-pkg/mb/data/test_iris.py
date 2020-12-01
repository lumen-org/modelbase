# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for the iris data set
"""

import pandas as pd
import os

from mb.modelbase import MixableCondGaussianModel

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"
_csvfilepath_disc = os.path.splitext(__file__)[0] + "_discretized.csv"


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


def mcg_map_model():
    return MixableCondGaussianModel("TestMod").fit(df=mixed(), fit_algo="map")


if __name__ == '__main__':
    pass