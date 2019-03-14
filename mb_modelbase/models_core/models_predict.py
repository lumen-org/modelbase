# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas

This module contains helper functions methods to do prediction on models.
"""
import functools
import logging

import numpy as np
import pandas as pd
import multiprocessing as mp
import multiprocessing_on_dill as mp_dill

from mb_modelbase.models_core import splitter as sp
from mb_modelbase.models_core.base import Split, Condition, Density
from mb_modelbase.models_core.base import NAME_IDX, METHOD_IDX, YIELDS_IDX, ARGS_IDX, OP_IDX, VALUE_IDX
from mb_modelbase.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
# form here ...
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def cartesian_product_multi(*dfs):
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))
# ... till here

def _crossjoin2(*dfs):
    return cartesian_product_multi(*dfs)


def _crossjoin(df1, df2):
    return pd.merge(df1, df2, on='__my_cross_index__', copy=False)


def _crossjoin3(*dfs):
    dfs = (pd.DataFrame(data=df).assign(__my_cross_index__=1) for df in dfs if not df.empty)
    try:
        initial = next(dfs)
    except StopIteration:
        return pd.DataFrame()
    else:
        return functools.reduce(_crossjoin, dfs, initial).drop('__my_cross_index__', axis=1)


def _tuple2str(tuple_):
    """Returns a string that summarizes the given split tuple or aggregation tuple

    The indended use if as column names in the returned DataFrame of Model.predict.
    """
    is_aggr_tuple = len(tuple_) == 4 and not tuple_[METHOD_IDX] == 'density'
    prefix = (str(tuple_[YIELDS_IDX]) + '@') if is_aggr_tuple else ""
    return prefix + str(tuple_[METHOD_IDX]) + '(' + str(tuple_[NAME_IDX]) + ')'


def type_of_clause(clause):
    """Return the type of given clause: 'split', 'maximum', 'average', 'probability' or 'density'."""
    if isinstance(clause, str):
        return 'split'
    else:
        return clause[METHOD_IDX]


def get_names_and_maps():
    pass


def add_split_for_defaulting_field(dim, splitby, split_names, where, filter_names):
    """ A add a filter and identity split for the defaulting field `dim`, if possible.

    TODO: fix documentation!

    Returns:
        the value/subset `dim` defaults to.
    """
    name = dim['name']

    if dim['default_value'] is not None:
        def_ = dim['default_value']
    elif dim['default_subset'] is not None:
        def_ = dim['default_subset']
    else:
        raise ValueError("Missing split-tuple for a split-field in predict: " + name)

    logger.info("using default for dim " + str(name) + " : " + str(def_))

    # add split
    split = Split(name, 'identity')
    splitby.append(split)
    split_names.append(name)

    # add condition
    condition = Condition(name, '==', def_)
    where.append(condition)
    filter_names.append(name)

    return def_


def create_data_structures_for_clauses(model, predict, where, splitby, evidence):
    """
    TODO: fix documentation!
    :param model:
    :param predict:
    :param where:
    :param splitby:
    :param evidence:
    :return:
    """

    # the following convention for naming of variables is used:
    #  * 'input' means we require values of that dimension as input
    #  * 'output' means values of that dimension will be the result of a prediction
    #  * 'split' means dimension that are split by in order to create 'input'
    #  * 'dims' means dimensions that are required to be included in the model
    #
    # furthermore:
    #  * 'names' hold names of dimensions (which are unique along dimensions of a model, but multiple FieldUsages
    #    of one dimension may be part of a predict query). There is the special case of names of the
    #    Density/Probability Field Usages, which are not (and cannot be) names of dimensions as such.
    #  * 'ids' hold unique ids (which are unique for that particular FieldUsage across this call to `.predict()`)
    #
    # note also that if anything is a list it is in the same order than the corresponding arguments to `.predict()`

    idgen = utils.linear_id_generator()

    # init utility data structures for the various clauses: splits, aggr-predicions, density-predicions, ...
    filter_names = [f[NAME_IDX] for f in where]

    # predict.* is about the dimensions and columns of the pd.DataFrame to be returned as query result
    predict_ids = []  # labels for columns in result data frames. splits/evidence is labelled by just their name, aggr have an id
    predict_names = []  # names of columns as to be returned. In correct order. For renaming of columns.

    # split.* is about the splits to to in order to generate necessary input
    split_names = [f[NAME_IDX] for f in splitby]  # name of fields to split by. Same order as in split-by clause.

    # aggr.* is about the aggregations to do
    aggrs = []  # list of aggregation tuples, in same order as in the predict-clause
    aggr_ids = []  # ids for columns of fields to aggregate. Same order as in predict-clause
    aggr_dims = []  # names of the dims averaged or maximized over
    aggr_input_names = set()  # names of the dims requires as input for density / probability
    aggr_output_names = []  # name of the dim predicted by each aggregation, in same order as aggr

    # evidence.* is about the data points to use as input
    evidence_names = evidence.columns

    # check / normalize each predict clause
    for clause in predict:
        normalize_predict_clause(model, clause, aggrs, aggr_ids, aggr_dims, aggr_input_names, aggr_output_names,
                             splitby, split_names,
                             where, filter_names,
                             predict_ids, predict_names,
                             evidence_names,
                             idgen)

    return aggrs, aggr_ids, aggr_input_names, aggr_dims, \
        predict_ids, predict_names, \
        split_names, \
        evidence, evidence_names


def normalize_predict_clause(model, clause, aggrs, aggr_ids, aggr_dims, aggr_input_names, aggr_output_names,
                             splitby, split_names, where, filter_names, predict_ids, predict_names,
                             evidence_names, idgen):
    """
    TODO: fix documentation!
    """
    clause_type = type_of_clause(clause)
    if clause_type == 'split':
        # t is a string, i.e. name of a field that is split by
        name = clause
        if name not in split_names:
            dim = model.byname(name)
            add_split_for_defaulting_field(dim, splitby, split_names, where, filter_names)
        predict_names.append(name)
        predict_ids.append(name)

    else:
        # t is an aggregation/density tuple
        clause_name = _tuple2str(clause)
        clause_id = clause_name + next(idgen)

        aggrs.append(clause)
        aggr_ids.append(clause_id)

        predict_names.append(clause_name)  # generate column name to return
        predict_ids.append(clause_id)

        if clause_type == 'maximum' or clause_type == 'average':
            aggr_dims.extend(clause[NAME_IDX])
            aggr_output_names.append(clause[YIELDS_IDX])

        elif clause_type == 'density' or clause_type == 'probability':
            input_names = clause[NAME_IDX]
            aggr_input_names.update(input_names)
            aggr_output_names.append(clause_name)

            # all input dimensions for a density/probability aggregation must be available:
            #  * as a split (i.e. splitby), or
            #  * from data (i.e. evidence)
            # if that is not the case: add a default split for it
            for name in input_names:
                if name not in evidence_names and name not in split_names:
                    add_split_for_defaulting_field(model.byname(name),
                                                   splitby, split_names,
                                                   where, filter_names)
        else:
            raise ValueError('invalid clause type: ' + str(clause_type))


def derive_aggregation_model(model, aggr, input_names, model_name=None):
    """Derive a model from model for the aggregation `aggr` considering that we have input along dimensions in
    `input_names`.

    TODO: fix documentation!

    Returns: mb_modelbase.Model
    """
    aggr_names = set(aggr[NAME_IDX])
    clause_type = type_of_clause(aggr)
    if clause_type == 'density' or clause_type == 'probability':
        assert (input_names >= aggr_names)
        dims_to_model = input_names
    else:
        assert set(input_names).isdisjoint(aggr_names)
        dims_to_model = input_names | aggr_names
    #TODO: ??? dims_to_model = model.sorted_names(dims_to_model)
    return model.copy(name=model_name).model(model=list(dims_to_model))


def get_split_values(model, split):
    field = model.byname(split[NAME_IDX])
    domain = field['domain'].bounded(field['extent'])
    try:
        splitfct = sp.splitter[split[METHOD_IDX].lower()]
    except KeyError:
        raise ValueError("split method '" + split[METHOD_IDX] + "' is not supported")
    return splitfct(domain.values(), split[ARGS_IDX])


def get_group_frame(model, split, column_id):
    frame = pd.DataFrame({column_id: get_split_values(model, split)})
    frame['__crossIdx__'] = 0  # need that index to cross join later
    return frame


def data_splits_to_evidence(model, splitby, evidence):
    """Convert data splits into evidence and return that evidence.

    Careful: splitby is not modified.
    """

    # find data_splits
    data_split_names = [s[NAME_IDX] for s in splitby if s[METHOD_IDX] == 'data']

    # extract test data for it
    if len(data_split_names) > 0:
        data_split_data = model.test_data.loc[:, data_split_names].sort_values(by=data_split_names, ascending=True)
        #TODO:!? data_split_data.columns = data_ids  # rename to data split ids!

        # add to evidence
        if evidence.empty:
            evidence = data_split_data
        elif evidence.columns == data_split_names:
            # may only add if column are identical
            evidence = pd.concat([evidence, data_split_data])
        else:
            raise ValueError("cannot merge evidence with data splits if dimensions and their order are not identical.")

    return evidence


def generate_all_input(model, input_names, splits, split_names, evidence):
    """Returns a tuple of two pd.DataFrames of input for the prediction query execution.

    The first is for partial data, i.e. data of (a subspace of) the data space that is used item-wise as input. That
    means columns of it are combined using concatenation.

    The second is for split data, i.e. data that is used column wise. That means columns of it are combined using
    cross-joins.

    The two data frames have no common columns.

    Args:
        model: md_modelbase.Model
        input_names: sequence of str
            The names of the dimensions to generate input series' for.

    Returns: pd.DataFrame, pd.DataFrame
        The generated dict of <input dimension name : series of input values>
    """

    # normalize splits with method 'data' to evidence
    evidence = data_splits_to_evidence(model, splits, evidence)
    assert set(evidence.columns).isdisjoint(set(input_names))

    # generate input series for each input name
    name2split = dict(zip(split_names, splits))
    data_dict = {name: generate_input_series_for_dim(model, name, split_names, name2split, evidence) for name in input_names}

    return evidence, data_dict


def generate_input_for_aggregation(model, aggr, split_data, partial_data):
    """Return an iterator over """
    pass


def generate_input_series_for_dim(model, input_dim_name, split_names, name2split, evidence):
    name = input_dim_name
    split_names = set(split_names)

    # if evidence is available: use evidence
    if name in evidence.columns:
        series = evidence.loc[:, name]

        # if split is available: modify evidence
        if name in split_names:
            raise NotImplementedError('tbd')

    # if no evidence is available but split: use split
    elif name in split_names:
        series = pd.Series(data=get_split_values(model, name2split[name]), name=name)

    # if also no split is available
    else:
        # but if there is a default: use default
        dim = model.byname(name)
        def_ = None

        if dim['default_value'] is not None:
            def_ = dim['default_value']
        elif dim['default_subset'] is not None:
            def_ = dim['default_subset']

        if def_ is not None:
            series = pd.Series(data=[def_], name=name)
        else:
            # raise error...
            raise ValueError('')

    assert(series is not None)
    return series


def generate_input_frame(model, basemodel, splitby, split_ids, evidence):
    """OLDOLDOLD
    Create one common big input based on splitbys"""

    # TODO: in the future I want to create the input frame based on what is actually needed for a particular aggregation
    # instead of just creating one huge input frame and then cutting down what is not needed
    # Bad Idea: instead change semantic, so that always all of the frame is needed

    # either something is needed as input for density/prob or it's used as a condition

    # TODO: idea: can't we handle the the split-method 'data' (which uses .test_data as the result of the split) as evidence
    # this seem more clean and versatile

    # TODO: we should only need basemodel

    if len(splitby) == 0:
        if evidence is not None:
            input_frame = evidence
        else:
            input_frame = pd.DataFrame()
    else:
        # filter to tuples of (identity_split, split_id)
        id_tpl = tuple(zip(*((s, i) for s, i in zip(splitby, split_ids) if s[METHOD_IDX] == 'identity')))
        identity_splits, identity_ids = ([], []) if len(id_tpl) == 0 else id_tpl

        # filter to tuples of (data_split, split_id)
        split_tpl = tuple(zip(*((s, i) for s, i in zip(splitby, split_ids) if s[METHOD_IDX] == 'data')))
        data_splits, data_ids = ([], []) if len(split_tpl) == 0 else split_tpl

        # all splits are non-data splits. Note: this should be the 'normal' case
        if len(data_splits) == 0:
            group_frames = map(get_group_frame, [model] * len(splitby), splitby, split_ids)
            input_frame = functools.reduce(_crossjoin, group_frames, next(group_frames)).drop('__crossIdx__', axis=1)

        # all splits are data and/or identity splits
        elif len(data_splits) + len(identity_splits) == len(splitby):

            # compute input frame according to data splits
            data_split_names = [s[NAME_IDX] for s in data_splits]
            assert (model.mode == 'both')
            # limit = 15*len(data_split_names)  # TODO: maybe we need a nicer heuristic? :)
            # #.drop_duplicates()\ # TODO: would make sense to do it, but then I run into problems with matching test data to aggregations on them in frontend, because I drop them for the aggregations, but not for test data select
            # sorting is required for correct display, since points are connected along their index
            input_frame = model.test_data.loc[:, data_split_names].sort_values(by=data_split_names, ascending=True)
            input_frame.columns = data_ids  # rename to data split ids!

            # add identity splits
            for id_, s in zip(identity_ids, identity_splits):
                field = basemodel.byname(s[NAME_IDX])
                domain = field['domain'].bounded(field['extent'])
                assert (domain.issingular())
                input_frame[id_] = domain.value()

            # TODO: I do not understand why this reset is necesary, but it breaks if I don't do it.
            input_frame = input_frame.reset_index(drop=True)
        else:
            raise NotImplementedError('Currently mixing data splits with any other splits is not supported.')

    return input_frame


def divide_df(df, colnames):
    """Returns a tuple of two pd.DataFrames where the first one contains all columns with names in `colnames` and the
    second all other.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame

    assert(set(df.columns) >= set(colnames))
    other = set(df.columns) - set(colnames)
    return df[list(colnames)], df[list(other)]


def aggregate_density_or_probability(model, aggr, partial_df, split_data_dict, aggr_id='aggr_id'):  # , name2id):
    """
    Compute density or probability aggregation `aggr` for `model` on given data.
    :param model:
    :param aggr:
    :param partial_df: pd.DataFrame
    :param split_data_dict: dict
    :param name2id:
    :return:
    """

    # data has two 'types' of dimension:
    #  * `input_names`: input to the density query
    #  * `cond_out_names`: to be conditioned out

    input_names = model.sorted_names(aggr[NAME_IDX])
    cond_out_names = (set(partial_df.columns) | set(split_data_dict.keys())) - set(input_names)

    # divide evidence and split data into those for conditioning out and for querying
    partial_df_cond_out, partial_df_input = divide_df(partial_df, cond_out_names)
    split_df_cond_out, split_df_input = [], []
    for name, df in split_data_dict.items():
        if name in cond_out_names:
            split_df_cond_out.append(df)
        else:
            split_df_input.append(df)

    # build data of both 'types'
    cond_out_data = _crossjoin3(*split_df_cond_out, partial_df_cond_out)
    input_data = _crossjoin3(*split_df_input, partial_df_input)

    # reorder to match model ordering
    input_data = input_data[model.sorted_names(input_data.columns)]

    # TODO: make the outer loop parallel
    operator_list = ['==']*len(input_names)  # OLD: used operator_list with custom op string

    results = []
    method = aggr[METHOD_IDX]
    if cond_out_data.empty:
        results = aggr_density_probability_inner(model, method, input_data)
    else:
        _extend = results.extend
        for row in cond_out_data.itertuples(index=False, name=None):
            # derive model for these specific conditions
            pairs = zip(input_names, operator_list, row)
            cond_out_model = model.copy().condition(pairs).marginalize(keep=input_names)
            # query model
            _extend(aggr_density_probability_inner(cond_out_model, method, input_data))

    # TODO: use multi indexes. this should give some speed up
    #  problem right now is: the columns for the keys are not hashable, because they contains lists
    #  solution: turn lists into tuples. Not sure how deep required changes would be
    # foo = _crossjoin3(cond_out_data, input_data)
    # df_index = pd.MultiIndex.from_frame(foo)
    # return_df = pd.DataFrame(data={aggr_id: results}, index=df_index)
    # return return_df

    return _crossjoin3(cond_out_data, input_data).assign(**{aggr_id: results})


def aggr_density_probability_inner(model, method, input_data):
    """Compute density/probability of given `model` w.r.t to  `input_data`.

    Args:
        model: mb_modelbase.Model
            The model to get density/probability for.
        method: str,
            One of 'density' and 'probability'
        input_data: pd.DataFrame
            data frame to use as input for model.
            Input data must be in same order like fields in model.

    Returns: list
        The probability/density values.
    """
    # TODO: is this still an issue?
    # # when splitting by elements or identity we get single element lists instead of scalars.
    # # However, density() requires scalars.
    # # TODO: I believe this issue should be handled in a conceptually better and faster way...
    # nonscalar_ids = [input_name2id[name] for (name, method, __) in splitby if
    #                  method == 'elements' or method == 'identity' and name in names]
    # for col_id in nonscalar_ids:
    #     subframe[col_id] = subframe[col_id].apply(lambda entry: entry[0])

    results = []
    if method == 'density':
        assert(model.names == list(input_data.columns))
        if model.parallel_processing:
            with mp.Pool() as p:
                results = p.map(model.density, input_data.itertuples(index=False, name=None))
        else:  # Non-parallel execution
            _density = model.density
            _append = results.append
            for row in input_data.itertuples(index=False, name=None):
                _append(_density(values=row))

    else:  # aggr_method == 'probability'
        assert (method == 'probability')
        if model.parallel_processing:
            with mp.Pool() as p:
                results = p.map(model.probability, input_data.itertuples(index=False, name=None))
        else:
            # TODO: use DataFrame.apply instead? What is faster?
            _probability = model.probability
            _append = results.append
            for row in input_data.itertuples(index=False, name=None):
                _append(_probability(domains=row))

    assert(len(input_data) == len(results))
    return results


def aggregate_maximum_or_average(model, aggr, partial_data, split_data_dict, input_names, splitby, aggr_id='aggr_id'):
    """
    TODO: fix documentation

    :param model:
    :param aggr:
    :param partial_data:
    :param split_data_dict:
    :param input_names:
    :param splitby:
    :param aggr_id:
    :return:
    """

    input_names = model.sorted_names(aggr[NAME_IDX])
    split_data_list = (df for name, df in split_data_dict.items())

    cond_out_data = _crossjoin3(*split_data_list, partial_data)
    cond_out_names = set(partial_data.columns) | set(split_data_dict.keys())

    # TODO: make the outer loop parallel
    operator_list = ['==']*len(input_names)  # OLD: used operator_list with custom op string
    # TODO: speed up results = np.empty(len(input_frame))
    results = []

    if len(cond_out_data) == 0:
        # there is no fields to split by, hence only a single value will be aggregated
        assert len(splitby) == 0
        # singlemodel = aggr_model.copy().marginalize(keep=aggr[NAME_IDX])
        assert len(aggr[NAME_IDX]) == len(model.fields)
        res = model.aggregate(aggr[METHOD_IDX], opts=aggr[ARGS_IDX + 1])
        i = model.asindex(aggr[YIELDS_IDX])  # reduce to requested field
        results.append(res[i])

    else:
        row_id_gen = utils.linear_id_generator(prefix="_row")
        rowmodel_name = model.name + next(row_id_gen)

        # TODO: I think this can be speed up: no need to recompute `i`, i is identical for all iteration of for loop

        def pred_max_func(row, cond_out_names=cond_out_names, operator_list=operator_list,
                                   rowmodel_name=rowmodel_name, model=model):
            pairs = zip(cond_out_names, operator_list, row)
            rowmodel = model.copy(name=rowmodel_name).condition(pairs).marginalize(keep=aggr[NAME_IDX])
            res = rowmodel.aggregate(aggr[METHOD_IDX], opts=aggr[ARGS_IDX + 1])
            i = rowmodel.asindex(aggr[YIELDS_IDX])
            return res[i]

        if model.parallel_processing:
            with mp_dill.Pool() as p:
                results = p.map(pred_max_func, cond_out_data.itertuples(index=False, name=None))
        else:
            results = [pred_max_func(row) for row in cond_out_data.itertuples(index=False, name=None)]

    return cond_out_data.assign(**{aggr_id: results})
