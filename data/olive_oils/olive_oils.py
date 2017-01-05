# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the olive oils data set
"""

import pandas as pd


def mixed(datafilepath='olive.csv', region_area_flag="region", ):
    """Loads the set from a csv file, does some useful preprocessing and returns the
    remaining data as a pandas data frame. Returns both, categorical and continuous data columns.

    Args:
        region_area_flag:
            set to "region" to keep only the "region" column, and remove the "area" column
            set to "area" to keep only the "area" column, and remove the "area" column
            set to "both" to keep both, the "region" and "area" column
    """
    df = pd.read_csv(datafilepath)

    # drop numerical area column
    del df['area-int']

    # apply flag
    if region_area_flag == 'region':
        del df['area']
    elif region_area_flag == 'area':
        del df['region-int']

    # if region is kept we translate it to proper levels
    if 'region-int' in df.columns:
        df['region'] = df['region-int'].astype('category').cat.rename_categories(["south", "sardinia", "north"])
        del df['region-int']

    return df


if __name__ == "__main__":
    df = mixed(region_area_flag="region")
