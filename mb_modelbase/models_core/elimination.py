# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas

"""

import numpy as np
from sklearn import metrics

import mb_modelbase.models_core.models
from mb_modelbase.models_core.base import *

def eliminate(model, df, elim_dims, **kwargs):
    """ Eliminate influence on data in one representation from certain dimensions in the other representation.

    Elimination provides a forward and backward mode.

    Forward mode: This means you eliminate the influence of original dimensions on the DR projection. It aims
    answer questions like:
     * 'How would the DR projection look like if there is no information on dimension X available?',
        (marginalization)
     * 'How would the DR projection look like if all values of X are set to value a?' (conditioning)

    Backward mode: Like forward mode but removes the influence of projected dimensions on the original data
    space.

    Note: self is the model.

    Args:
        model : Model
            The model to use for elimination
        df : Pandas DataFrame
            The data from which to eliminate influence
        elim_dims : single or sequence of scalar conditions or dimension names.
            The dimensions to eliminate, either using marginalization or conditioning on a common value.
            If dimension names are given, marginalization is used. If conditions are given, conditining is
            used.

    Return:
        A dataframe with identical shape like `df` but with any influence of `elim_dims` removed.

    Raises:
        ValueError :
            If `df` and `elim_dims` share any dimensions, or
            if `elim_dims` holds conditions, but not conditioning on a single value.
    """

    # normalize elim_dims to list of scalar conditions
    # assert there are no overlapping dimensions

    # check: is it forward or backward elimination
    # check: is it marginalization or conditioning

    # retrieve other representation for items in df
    df_alternate = TODO
    # if marginalization: remove elim_dims columns
    # if conditioning: set elim_dims to provided values

    # predict dimensions of df for each of the
    influence = TODO

    # substract influence
    return df - influence


def rmse(model, input_df, groundtruth_df):

    predict_names = list(groundtruth_df.columns)

    if 0 != len(set(input_df.columns) & set(predict_names)):
        raise ValueError("colums of given dataframes may not overlap")

    predict_aggrs = [Aggregation(predict_names, yields=name) for name in predict_names]  # construct all Aggregations

    # pred = model.predict(predict=predict_aggrs, for_data=input_df)

    preds = []
    for name in predict_names:
        pred = model.predict(predict=Aggregation(predict_names, yields=name), for_data=input_df)
        preds.append(pred)
    print(preds)

    return metrics.mean_squared_error(y_true=groundtruth_df.values, y_pred=preds)


if __name__ == '__main__':
    import pandas as pd
    from sklearn import metrics

    # load conditional gaussian model on iris data
    m = mb_modelbase.Model.load('/home/luca_ph/Documents/projects/graphical_models/code/data_models/mcg_iris_map.mdl')

    data = m.data

    # prediction of categorical species
    # data_evidence = data.iloc[:, 1:]
    # data_ground_truth = data.iloc[:, 0]
    # data_prediction = m.predict(Aggregation('species'), for_data=data_evidence)
    #
    # cm = metrics.confusion_matrix(data_ground_truth, data_prediction, labels=data_ground_truth.unique())
    # print('confusion matrix:\n{}'.format(cm))

    # prediction of some quantitative dim
    # err = rmse(m, data.iloc[:,:-1], data.iloc[:,-1:] )
    # print('rsme (1dim):\n{}'.format(err))

    m.parallel_processing = False

    err = rmse(m, data.iloc[:,:-2], data.iloc[:,-2:] )
    print('rsme (2dim):\n{}'.format(err))
