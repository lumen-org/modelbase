# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the diamonds data set.
Data set is taken from the ggplot2 package from the R language.
"""

import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"


def mixed(filepath=_csvfilepath):
    """Loads the diamonds data set from a csv file, removes the index column and returns the
    remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath)

    # drop depth column, it simply is:  2 * z / (x + y)
    del df['depth']

    return df


def mixed_cat3(filepath=_csvfilepath):

    df = pd.read_csv(filepath)
    df = pd.DataFrame(df)

    del df['depth']

    cut = {"Fair": "Good", "Very Good": "Premium"}
    df.cut.replace(to_replace=cut, inplace=True)

    col = {"J": "Bad", "I": "Bad", "H": "Medium", "G": "Medium",
           "F": "Medium", "E": "Excellent", "D": "Excellent"}
    df.color.replace(to_replace=col, inplace=True)

    clar = {"I1": "Bad", "SI2": "Bad", "SI1": "Bad", "VS2": "Medium",
            "VS1": "Medium", "VVS2": "Medium", "VVS1": "Excellent", "IF": "Excellent"}
    df.clarity.replace(to_replace=clar, inplace=True)

    # Drop not suitable rows
    df = df.drop(df[df.y == 0.00].index)
    df = df.drop(df[df.y > 15].index)
    df = df.drop(df[df.x == 0.00].index)
    df = df.drop(df[df.x > 15].index)
    df = df.drop(df[df.z == 0.00].index)
    df = df.drop(df[df.z > 15].index)

    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    return df
