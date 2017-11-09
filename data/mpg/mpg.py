"""
@author: Philipp Lucas

This data set is taken from: https://github.com/hadley/data-fuel-economy
"""
import pandas as pd
import os


_filepath = os.path.join(os.path.dirname(__file__), 'mpg.csv')

# that should about be all useful variables:
cols = ['year', 'class', 'displ', 'cyl', 'cty', 'hwy', 'drv', 'trans']

# temporal variables:
temp_cols = ['year']
# year is first year when car has been sold

# numerical variables
num_cols = ['displ', 'cyl', 'cty', 'hwy']
# displ is displacement of engine
# cyl is number of cylinders
# cty is miles per gallon in city
# hwy is miles per gallon on highway

# categorical variables:
cat_cols = ['class', 'drv', 'trans']
# class is ""compact car" and such
# drv is RWD or FWD
# trans is transmission type


def cg(file=_filepath):
    mpgdf = pd.read_csv(file)
    cols = ['year', 'class', 'cyl', 'displ', 'cty', 'hwy']
    mpgdf = mpgdf.loc[:, cols]
    mpgdf.drop(mpgdf.columns[[0]], axis=1, inplace=True)
    mpgdf.dropna(axis=0, inplace=True, how="any")
    return mpgdf

def cg2(file=_filepath):
    mpgdf = pd.read_csv(file)
    cols_cat = ['year', 'class', ]
    cols_num = ['displ', 'cyl', 'cty', 'hwy', ]
    cols = cols_cat + cols_num
    mpgdf = mpgdf.loc[:, cols]
    mpgdf['year'] = mpgdf['year'].astype('category')
    mpgdf.dropna(axis=0, inplace=True, how='any')  # drops rows with any NaNs
    return mpgdf

def cg_generic(file=_filepath, cols=cat_cols+num_cols):
    df = pd.read_csv(file)
    df = df.loc[:, cols]
    df.dropna(axis=0, inplace=True, how="any")
    return df

if __name__ == "__main__":
    df = cg()
    pass