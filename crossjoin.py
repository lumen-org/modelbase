"""
The following is taken from: 
https://gist.github.com/internaut/5a653317688b14fd0fc67214c1352831#file-pandas_crossjoin_example-py

Date: 2016-08-04


Shows how to do a cross join (i.e. cartesian product) between two pandas DataFrames using an example on
calculating the distances between origin and destination cities.

Tested with pandas 0.17.1 and 0.18 on Python 3.4 and Python 3.5

Best run this with Spyder (see https://github.com/spyder-ide/spyder)
Author: Markus Konrad <post@mkonrad.net>

April 2016
"""

import pandas as pd

def crossjoin(df1, df2, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    NOT ANYMORE: Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    INSTEAD: resets the index
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    
    if df1.size == 0:
        return df2
    if df2.size == 0:
        return df1
    
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)#.reset_index()
    #res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res