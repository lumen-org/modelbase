# Copyright (c) 2020 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas
"""

import pandas as pd
import mb_modelbase as mb


def repredict(model, data, include=None):
    """Predicts attribute values of data given all but the predicted attributes.

    Arguments:
        model: Model
            A compatible model for prediction.
        data: pd.DataFrame
            A data frame of data to repredict and use as conditions for predictions.
        include: list of strings, optional.
            List of dimensions to re-predict. If specified, only the included dimensions will be
            predicted.

    Returns: pd.DataFrame
        DataFrame of predictions.
    """
    # marginalize to smallest required model
    dim_names = list(data.columns)
    model = model.copy().marginalize(keep=dim_names)
    model.mode = 'model'
    predict_names = include if include is not None else dim_names

    # repredict all dimensions
    repredict_series = [model.predict([mb.Aggregation(dim_name)], for_data=data.loc[:, data.columns != dim_name])
                        for dim_name in predict_names]

    # combine to one data frame
    repredict_df = pd.concat(repredict_series, axis=1)
    repredict_df.columns = predict_names
    return repredict_df


def repredict_data_difference(model, data, reprediction=None, numerical_diff='difference',
                              string_diff='equality', **kwargs):
    """"Calculates the element-wise difference between data and its repredictions with respect to a
    model.

    For difference the following measures are used:
        categorical dimension: element-wise '=='
        quantitative dimension: element-wise data - reprediction

    Arguments:
        model: mb.Model
        data: pd.DataFrame
        reprediction: pd.DataFrame, optional.
            If given, this dataframe is used as reprediction. Otherwise, it is generated using
            `repredict`.
        numerical_diff: string
            one of diffMethods['numerical'].keys()
        string_diff: string
            one of diffMethods['string'].keys():
        kwargs: These arguments are passed on to `repredict`.

    Returns: pd.DataFrame
        A data frame of identical shape and column names as reprediction. It contains element-wise
        difference between original data and reprediction.
    """
    if string_diff not in diffMethods['string'].keys():
        raise ValueError("invalid comparision for string dtype: {}".format(string_diff))
    if numerical_diff not in diffMethods['numerical'].keys():
        raise ValueError("invalid comparision for numerical dtype: {}".format(numerical_diff))
    diff_method = {
        'string': diffMethods['string'][string_diff],
        'numerical': diffMethods['numerical'][numerical_diff],
    }

    if reprediction is None:
        reprediction = repredict(model, data, **kwargs)

    reprediction_dim_names = list(reprediction.columns)
    reprediction_dtype = model.dtypes(reprediction_dim_names)

    diff_series = []
    for dim_name, dtype in zip(reprediction_dim_names, reprediction_dtype):
        diff_df = diff_method[dtype](reprediction[dim_name], data[dim_name])
        diff_series.append(diff_df)
    return pd.concat(diff_series, axis=1)


def diff_numerical_rsme(df1, df2):
    diff = diff_numerical_difference(df1, df2)
    return (diff ** 2).mean() ** .5


def diff_numerical_absolute_difference(df1, df2):
    return diff_numerical_difference(df1, df2).abs()


def diff_numerical_difference(df1, df2):
    return df1.sub(df2, axis='rows')


def diff_string_equality(df1, df2):
    return df1.eq(df2, axis='rows')


diffMethods = {
    'string': {
        'equality': diff_string_equality,
    },
    'numerical': {
        'difference': diff_numerical_difference,
        'absolute_difference': diff_numerical_absolute_difference
    }
}

if __name__ == '__main__':
    pass
