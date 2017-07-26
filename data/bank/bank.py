# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the bank data set.
"""

import pandas as pd

def mixed(datafilepath='data/bank/bank-full.csv'):
    """Loads the set from a csv file, does some useful preprocessing and returns the
    remaining data as a pandas data frame.
    """

    # TODO: not so sure if this is working!

    # handy shortcut: close all open figures
    #plt.close("all")

    # data preperation of the adults data set for use in a purely categorical model

    # open data file

    df = pd.read_csv(datafilepath, delimiter=";", index_col=False, skipinitialspace=True)

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


if __name__ == "__main__":
    df = mixed()
   
   
   
   
   
   
   

