# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the crabs data set
"""

import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"


def mixed(filepath=_csvfilepath):
    """Loads the crabs data set from a csv file, removes the index column and returns the
    remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath)

    # drop index column
    del df['index']

    return df


def continuous(filepath=_csvfilepath):
    df = mixed(filepath)
    del df['sex']
    del df['species']
    return df


def categorical(filepath=_csvfilepath):
    df = mixed(filepath)
    return df[['sex','species']]
