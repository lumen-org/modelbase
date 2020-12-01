# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for some synthetic data
"""

import pandas as pd
import os

_csvfilepath_categorical = os.path.splitext(__file__)[0] + "_categorical.csv"
_csvfilepath_mixed = os.path.splitext(__file__)[0] + "_mixed.csv"


def categorical(filepath=_csvfilepath_categorical):
    """Loads a synthetic categorical data set and returns the data as a pandas data frame
    """
    df = pd.read_csv(filepath)
    return df


def mixed(filepath=_csvfilepath_mixed):
    """Loads a synthetic mixed typed data set and returns the data as a pandas data frame
    """
    df = pd.read_csv(filepath)
    return df
