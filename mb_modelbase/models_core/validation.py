
import pandas as pd
import numpy as np
from sklearn import metrics

from mb_modelbase.models_core.base import *

# TODO: see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

def confusion_matrix(model, input_df, groundtruth_series):
    """Returns the predicted values and their confusion matrix on given input and groundtruth using model.
    """
    # TODO: assert that groundtruth_series is categorical
    data_prediction = model.predict(Aggregation('species'), for_data=input_df)
    return data_prediction, metrics.confusion_matrix(groundtruth_series, data_prediction, labels=groundtruth_series.unique())


def rmse(model, input_df, groundtruth_df):

    predict_names = list(groundtruth_df.columns)

    if 0 != len(set(input_df.columns) & set(predict_names)):
        raise ValueError("colums of given dataframes may not overlap")

    predict_aggrs = [Aggregation(predict_names, yields=name) for name in predict_names]  # construct all Aggregations

    preds = model.predict(predict=predict_aggrs, for_data=input_df)

    # preds = []
    # for name in predict_names:
    #     pred = model.predict(predict=Aggregation(predict_names, yields=name), for_data=input_df)
    #     preds.append(pred)
    # print(preds)

    return preds, metrics.mean_squared_error(y_true=groundtruth_df.values, y_pred=preds.values,multioutput='raw_values')
