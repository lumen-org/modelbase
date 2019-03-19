# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas

"""

import mb_modelbase.models_core.models


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


def data_predict(model, predict, for_data, **kwargs):
    """

    In `.eliminate` it turned our that we need a prediction method that takes data items as input which will be used
    as evidence for prediction. This is indeed a very common operation, e.g. imagine the iris data set where we want
    to check the quality of the model and hence compare the predicted value of 'species' with the actual one. Then,
    for each data item we need to predict the species, given the known values of the other dimensions.

    We want to be able to do it like this:

    data_predict(self, what='species', for_data=<df>)

    The `for_data` arguments really is just a different, more convinient way of providing list of list of
    conditions, where each element of the other list is the sequence of conditions to hold.

    Probably this would be the naive implementation. And really, for the gauss-based models I don't really
    see a better option anyway.

    TODO: Is it better to integrate it into existing `.predict()` or write a new class method?
    """

    if not isinstance(model, mb_modelbase.Model):
        raise TypeError("model is not of type mb_modelbase.Model .")

    if isinstance(predict, (str, tuple)):
        predict = [predict]
    if not model.isfieldname(predict):
        raise ValueError("predict is not a single or sequence of dimension names of model")
    data_dim_names = for_data.colnames
    if not model.isfieldname(data_dim_names):
        raise ValueError("for_data has data dimensions that are not modelled by model")

    # marginalize model to the dims in `predict` and `for_data`
    base_names = data_dim_names + predict
    base_model = model.marginalize(keep=base_names)

    # predict
    for item in for_data.itertuple()
        pass

    pass


if __name__ == '__main__':
    import pandas as pd
    from sklearn import metrics

    # load conditional gaussian model on iris data
    m = mb_modelbase.Model.load('/home/luca_ph/Documents/projects/graphical_models/code/data_models/mcg_iris_map.mdl')

    data = m.data
    data_evidence = data.iloc[:, 1:]
    print(data_evidence.head())

    # predict species from all other attributes
    data_prediction = data_predict(m, 'species', data_evidence)

    # count the number of false prediction
    data_ground_truth = data.iloc[:,0]

    # perf = pd.DataFrame(
    #     data={'species': data['species'],
    #           'predicted': data_prediction,
    #           'ground_truth': data_ground_truth
    #           })

    cm = metrics.confusion_matrix(data_ground_truth, data_prediction, labels=data_ground_truth.unique())
    print(cm)
