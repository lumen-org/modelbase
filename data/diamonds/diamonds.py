# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the diamonds data set.
Data set is taken from the ggplot2 package from the R language.
"""

import pandas as pd


def mixed(filepath='diamonds.csv'):
    """Loads the crabs data set from a csv file, removes the index column and returns the
    remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath)

    # drop depth column, it simply is:  2 * z / (x + y)
    del df['depth']

    return df
