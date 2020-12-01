# Copyright © 2016 Felix Glaser (felix.glaser@uni-jena.de)
"""
@author: Felix Glaser

Data Preprocessing and cleansing for the glass data set
"""

import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".data.csv"


def mixed(datafilepath=_csvfilepath):
    """Loads the set from a csv file, does some useful preprocessing and returns the
    remaining data as a pandas data frame. Returns both, categorical and continuous data columns.

    Args:
        none
    """
    #df = pd.read_csv(datafilepath, dtype={'type-int': 'category'})
    df = pd.read_csv(datafilepath, dtype={'type-int': 'object'})

    # drop id column
    del df['id']

    # translate type to proper levels
    df['type'] = df['type-int'].astype('category').cat.rename_categories(["bwfp", "bwnfp", "vwfp", "co", "tw", "hl"])
    del df['type-int']

    return df

if __name__ == "__main__":
    df = mixed()
