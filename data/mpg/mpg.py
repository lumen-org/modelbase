"""
@author: Kazimir Menzel, Philipp Lucas

This data set is taken from: https://github.com/hadley/data-fuel-economy
"""
import pandas as pd
import os

_filepath = os.path.join(os.path.dirname(__file__), 'mpg.csv')


def cg(file=_filepath):
    mpgdf = pd.read_csv(file)
    cols = ['year', 'class', 'cyl', 'displ', 'cty', 'hwy']
    mpgdf = mpgdf.loc[:, cols]
    mpgdf.drop(mpgdf.columns[[0]], axis=1, inplace=True)
    mpgdf.dropna(axis=0, inplace=True)
    return mpgdf


if __name__ == "__main__":
    df = cg()
    pass