# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for the car crashes data set
"""
#
import pandas as pd


def mixed(filepath='car_crashes.csv'):
    """Loads the car_crashes data set from a csv file, removes the index column and returns the remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath, index=False)
    return df


def continuous(filepath='car_crashes.csv'):
    df = mixed(filepath)
    del df['species']
    return df
