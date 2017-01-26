# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the yeast data set
"""

import pandas as pd


def mixed(datafilepath='yeast.csv'):
    """Loads the yeast data set from a csv file, and returns both, categorical and continuous
    remaining data as a pandas data frame
    """
    df = pd.read_csv(datafilepath)

    # drop name column, because it's an index
    del df['name']

    # categorize erl column: it's either 1.00 or 0.50
    df['erl'] = df['erl'].astype('category').cat.rename_categories(["low", "high"])

    # categorize pox column: it's either 0.00 or 0.50 or 0.83
    df['pox'] = df['pox'].astype('category').cat.rename_categories(["low", "medium", "high"])

    return df

if __name__ == "__main__":
    df = mixed()