# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for the car crashes data set
"""
#
import pandas as pd
import os

_filepath = os.path.join(os.path.dirname(__file__), 'car_crashes.csv')

def mixed(filepath=_filepath):
    """Loads the car_crashes data set from a csv file, removes the index column and returns the remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath)
    return df


def continuous(filepath=_filepath):
    df = mixed(filepath)
    del df['abbrev']
    return df
