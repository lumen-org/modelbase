# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for the iris data set
"""

import pandas as pd
import os

_filepath = os.path.join(os.path.dirname(__file__), 'iris.csv')


def mixed(filepath=_filepath):
    """Loads the iris data set from a csv file, removes the index column and returns the
    remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath)
    return df


def continuous(filepath=_filepath):
    df = mixed(filepath)
    del df['species']
    return df
