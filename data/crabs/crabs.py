# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the crabs data set
"""

import pandas as pd


def mixed():
    """Loads the crabs data set from a csv file, removes the index column and returns the
    remaining data as a pandas data frame
    """
    df = pd.read_csv('data/crabs/australian-crabs.csv')

    # drop index column
    del df['index']

    return df
