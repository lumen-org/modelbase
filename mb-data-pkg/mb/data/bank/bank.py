# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the bank data set.
"""

import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + "-full.csv"


def mixed(datafilepath=_csvfilepath):
    """Loads the set from a csv file, does some useful preprocessing and returns the
    remaining data as a pandas data frame.
    """

    # TODO: not so sure if this is working!

    # handy shortcut: close all open figures
    # plt.close("all")

    # data preperation of the adults data set for use in a purely categorical model

    # open data file

    df = pd.read_csv(datafilepath, delimiter=";",
                     index_col=False, skipinitialspace=True)

    # drop NA/NaN
    #datafilepath = 'data/adult/adult.full'
    #df = pd.read_csv(datafilepath, index_col=False, na_values='?')
    #dfclean = df.dropna()

    # print information about columns:
    print("Columns and their data type:")
    for col in df.columns:
        #print(df[col].name, " (", df[col].dtype, "), values counts: ", df[col].value_counts())
        print(df[col].name, " (", df[col].dtype)

    # print histograms on continuous columns
    df.hist()
    return df


def mixed_4cat(datafilepath=_csvfilepath):

    df = pd.read_csv(datafilepath, delimiter=";",
                     index_col=False, skipinitialspace=True)
    df = pd.DataFrame(df, index=None)

    del df['default']
    del df['contact']
    del df['month']
    del df['day']
    del df['pdays']
    del df['job']
    del df['poutcome']
    del df['marital']
    
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    return df
