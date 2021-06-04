# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data Preprocessing and cleansing for the bank data set.
"""

import pandas as pd
import os
import spn.structure.leaves.parametric.Parametric as spn_parameter_types
import spn.structure.StatisticalTypes as spn_statistical_types

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
    # datafilepath = 'data/adult/adult.full'
    # df = pd.read_csv(datafilepath, index_col=False, na_values='?')
    # dfclean = df.dropna()

    # print information about columns:
    print("Columns and their data type:")
    for col in df.columns:
        # print(df[col].name, " (", df[col].dtype, "), values counts: ", df[col].value_counts())
        print(df[col].name, " (", df[col].dtype)

    # print histograms on continuous columns
    df.hist()
    return df


def spn_parameter_mixed():
    bank_variable_types = {'age': spn_parameter_types.Gaussian, 'job': spn_parameter_types.Categorical,
                           'marital': spn_parameter_types.Categorical, 'education': spn_parameter_types.Categorical,
                           'default': spn_parameter_types.Categorical, 'balance': spn_parameter_types.Gaussian,
                           'housing': spn_parameter_types.Categorical, 'loan': spn_parameter_types.Categorical,
                           'contact': spn_parameter_types.Categorical, 'day': spn_parameter_types.Gaussian,
                           'month': spn_parameter_types.Categorical, 'duration': spn_parameter_types.Gaussian,
                           'campaign': spn_parameter_types.Gaussian, 'pdays': spn_parameter_types.Gaussian,
                           'previous': spn_parameter_types.Gaussian, 'poutcome': spn_parameter_types.Categorical,
                           'y': spn_parameter_types.Categorical}
    return bank_variable_types


def spn_metaparameters():
    bank_meta_types = {'age': spn_statistical_types.MetaType.REAL,
                       'job': spn_statistical_types.MetaType.DISCRETE,
                       'marital': spn_statistical_types.MetaType.DISCRETE,
                       'education': spn_statistical_types.MetaType.DISCRETE,
                       'default': spn_statistical_types.MetaType.DISCRETE,
                       'balance': spn_statistical_types.MetaType.REAL,
                       'housing': spn_statistical_types.MetaType.DISCRETE,
                       'loan': spn_statistical_types.MetaType.DISCRETE,
                       'contact': spn_statistical_types.MetaType.DISCRETE,
                       'day': spn_statistical_types.MetaType.REAL,
                       'month': spn_statistical_types.MetaType.DISCRETE,
                       'duration': spn_statistical_types.MetaType.REAL,
                       'campaign': spn_statistical_types.MetaType.REAL,
                       'pdays': spn_statistical_types.MetaType.REAL,
                       'previous': spn_statistical_types.MetaType.REAL,
                       'poutcome': spn_statistical_types.MetaType.DISCRETE,
                       'y': spn_statistical_types.MetaType.DISCRETE}
    return bank_meta_types


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

