# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
# Copyright (c) 2017 Kazimir Menzel (Kazimir.Menzel@uni-jena.de)
"""
@author: Philipp Lucas, Kazimir Menzel

Data Preprocessing and cleansing for the bank notes data set.

Origin of data set: https://archive.ics.uci.edu/ml/datasets/banknote+authentication
"""

import pandas as pd

def mixed(datafilepath='banknotes.csv'):
    """Loads the set from a csv file, does some useful preprocessing and returns the
    remaining data as a pandas data frame. Returns both, categorical and continuous data columns.
    """

    var_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(datafilepath, names=var_names)

    df['class'] = df['class'].astype('category')

    return df


if __name__ == "__main__":
    df = mixed()
    print(df.head())




