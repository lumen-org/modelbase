"""
@author: Kazimir Menzel, Philipp Lucas

This data set is taken from: https://github.com/hadley/data-fuel-economy
"""
import pandas as pd


def cg(file='data/mpg/mpg.csv'):
    mpgdf = pd.read_csv(file)
    cols = ['year', 'class', 'cyl', 'displ', 'cty', 'hwy']  # that was missing! :-)
    mpgdf = mpgdf.loc[:, cols]
    mpgdf.drop(mpgdf.columns[[0]], axis=1, inplace=True)
    mpgdf.dropna(axis=0, inplace=True)
    return mpgdf


if __name__ == "__main__":
    df = cg("mpg.csv")
