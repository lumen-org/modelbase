# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas

This module contains functions and methods to help do prediction on models with Model.predict.
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


def cartesian_product(*arrays):
    """Compute cartesian product

    Taken from here: https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def crossjoin2(*dfs):
    """Compute cartesian product

    Taken from here: https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
    """
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    return pd.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))


def _crossjoin_binary(df1, df2, join_on='__my_cross_index__'):
    """Computes the cartesian product of the two pd.DataFrames.

    Note, that the data frames are joined on a 'utility' column, which must exist in both data frames.
    """
    return pd.merge(df1, df2, on=join_on, copy=False)


def crossjoin(*dfs):
    """Compute cartesian product of all pd.DataFrame provided.
    """
    dfs = (pd.DataFrame(data=df).assign(__my_cross_index__=1) for df in dfs if not df.empty)
    try:
        initial = next(dfs)
    except StopIteration:
        return pd.DataFrame()
    else:
        return functools.reduce(_crossjoin_binary, dfs, initial).drop('__my_cross_index__', axis=1)


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


def add_split_for_defaulting_field(dim, splitby, split_names, where, filter_names):
    """ A add a identity split and a condition for the defaulting field `dim`, if possible.

    Args:
        dim: Field
            The defaulting dimension.
        splityby: [SplitTuple]
            List of splits.
        split_names: [str]
            List of names of the splits in splitby.
        where: [ConditionTuple]
            List of conditions.
        filter_names: [str]
            List of names of conditions.

    Returns:
        the value/subset `dim` defaults to.
    """
    name = dim['name']

    if dim['default_subset'] is not None:
        def_ = dim['default_subset']
        op_ = "in"
    elif dim['default_value'] is not None:
        def_ = dim['default_value']
        op_ = "=="
    else:
        raise ValueError("Missing split-tuple for a split-field in predict: " + name)

    logger.info("using default for dim " + str(name) + " : " + str(def_))

    # add split
    split = Split(name, 'identity')
    splitby.append(split)
    split_names.append(name)

    # add condition
    condition = Condition(name, op_, def_)
    where.append(condition)
    filter_names.append(name)

    return def_


def create_data_structures_for_clauses(model, predict, where, splitby, partial_data):
    """ Derive utility data structures from clauses of a predict query.

    General note: If anything is a list it is in the same order than the corresponding arguments to `.predict()`

    Args:
        model: Model
            Model that the predict query is run on.
        predict: []
            List of 'things' to predict. See header of Model.predict.
        where: [ConditionTuple]
            List of conditions. See header of Model.predict.
        splitby: [SplitTuple]
            List of splits. See header of Model.predict.
        partial_data: pd.DataFrame
            Partial data provided. See header of Model.predict.

    Return: (aggrs, aggr_ids, aggr_input_names, aggr_dims, predict_ids, predict_names, split_names, name2split,
           partial_data, partial_data_names)
        A long list of utility data structures. See inline documentation for an explanation of the names.
    """
    idgen = utils.linear_id_generator()
    filter_names = [f[NAME_IDX] for f in where]

    # predict.* is about the dimensions and columns of the pd.DataFrame to be returned as query result
    # labels for columns in result data frames. splits/partial_data is labelled by just their name, aggr have an id
    predict_ids = []
    # names of columns as to be returned, in same order. For renaming of columns before returning the final data frame.
    predict_names = []

    # split.* is about the splits to generate necessary input
    split_names = [f[NAME_IDX] for f in splitby]  # name of fields to split by. Same order as in split-by clause.

    # aggr.* is about the aggregations to do
    aggrs = []  # list of aggregation tuples, in same order as in the predict-clause
    aggr_ids = []  # ids for columns of fields that are aggregated. Same order as in predict-clause
    aggr_output_names = []  # name of the dims predicted by each aggregation, in same order as aggr

    aggr_dims = []  # names of the dims averaged or maximized over (not density/probability!)
    aggr_input_names = set()  # names of the dims requires as input for density / probability (not average/max!)

    # partial_data.* is about the user provided data points to use as input
    partial_data_names = partial_data.columns

    # check / normalize each predict clause
    for clause in predict:
        normalize_predict_clause(model, clause, aggrs, aggr_ids, aggr_dims, aggr_input_names, aggr_output_names,
                             splitby, split_names,
                             where, filter_names,
                             predict_ids, predict_names,
                             partial_data_names,
                             idgen)

    # maps name of splits to splits and meta data
    name2split = make_name2split(model, splitby)

    return aggrs, aggr_ids, aggr_input_names, aggr_dims, \
           predict_ids, predict_names, \
           split_names, name2split, \
           partial_data, partial_data_names


def make_name2split(model, splitby):
    """Make map of split name to split with meta data.

    Args:
        model: Model
            A model.
        splitby: [SplitTuple]
            List of splits to map.

    Returns: dict
    """
    name2split = {}
    for s in splitby:
        name = s[NAME_IDX]
        method = s[METHOD_IDX]
        input_type = model.byname(name)['dtype']
        return_type = sp.return_types[method]
        assert input_type in ['numerical', 'string'], 'invalid dtype'
        if input_type == 'numerical' and return_type == 'scalar':
            down_cast_fct = None
            up_cast_fct = None
        elif input_type == 'numerical' and return_type == 'domain':
            down_cast_fct = lambda x: (x[0] + x[1]) / 2
            up_cast_fct = None
        elif input_type == 'string' and return_type == 'scalar':
            down_cast_fct = None
            up_cast_fct = lambda x: (x,)
        elif input_type == 'string' and return_type == 'domain':
            down_cast_fct = lambda x: x[0]
            up_cast_fct = None

        name2split[name] = {
            'split': s,
            'method': method,
            'return_type': return_type,
            'cond_op': 'in' if return_type == 'domain' else '==',
            'down_cast_fct': down_cast_fct,
            'up_cast_fct': up_cast_fct
        }
    return name2split


def normalize_predict_clause(model, clause, aggrs, aggr_ids, aggr_dims, aggr_input_names, aggr_output_names,
                             splitby, split_names, where, filter_names, predict_ids, predict_names,
                             partial_data_names, idgen):
    """Normalizes the arguments given as 'clauses' to Model.predict.

    In particular it:
     * detects missing `Split`s. A typcial mistake when a query is issued.
     * generates explicit `Split`s for defaulting dimensions.
     * derives the values for (note that these are not returned but changed in place):
        predict_ids, predict_names, aggrs, aggr_ids aggr, aggr_output_names, aggr_dims, aggr_input_names
    """
    clause_type = type_of_clause(clause)
    if clause_type == 'split':
        # t is a string, i.e. name of a field that is split by
        name = clause
        if name not in split_names and name not in partial_data_names:
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
            #  * from data (i.e. partial data)
            # if that is not the case: add a default split for it
            for name in input_names:
                if name not in partial_data_names and name not in split_names:
                    add_split_for_defaulting_field(model.byname(name),
                                                   splitby, split_names,
                                                   where, filter_names)
        else:
            raise ValueError('invalid clause type: ' + str(clause_type))


def derive_aggregation_model(model, aggr, input_names, model_name=None):
    """Derive a model from model for the aggregation `aggr` considering that we have input along dimensions in
    `input_names`.

    Args:
        model: mb_modelbase.Model
        aggr: AggregationTuple
        input_names: set(str)
        model_name: str, optional.

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
    """Returns the values of `split` executed on `model`.

    Args:
        model: mb_modelbase.Model
        split: SplitTuple
    Returns: list
        List with values of the applied split.
    """
    field = model.byname(split[NAME_IDX])
    domain = field['domain'].bounded(field['extent'])
    try:
        splitfct = sp.splitter[split[METHOD_IDX].lower()]
    except KeyError:
        raise ValueError("split method '" + split[METHOD_IDX] + "' is not supported")
    return splitfct(domain.values(), split[ARGS_IDX])


def data_splits_to_partial_data(model, splitby, partial_data):
    """Convert data splits into partial_data and return that partial_data.

    Note: splitby is not modified, i.e. data splits are NOT removed.

    Args:
        model: mb_modelbase.Model
        splityby: [SplitTuple]
        partial_data: pd.DataFrame

    Returns: pd.DataFrame
        The (possibly) modified partial data.
    """

    # find data_splits
    data_split_names = [s[NAME_IDX] for s in splitby if s[METHOD_IDX] == 'data']

    # extract test data for it
    if len(data_split_names) > 0:
        data_split_data = model.test_data.loc[:, data_split_names].sort_values(by=data_split_names, ascending=True)
        #TODO:!? data_split_data.columns = data_ids  # rename to data split ids!
        # add to partial_data
        if partial_data.empty:
            partial_data = data_split_data
        elif partial_data.columns == data_split_names:
            # may only add if column are identical
            partial_data = pd.concat([partial_data, data_split_data])
        else:
            raise ValueError("cannot merge partial_data with data splits if dimensions and their order are not identical.")

    return partial_data


def generate_all_input(model, splits, split_names, partial_data):
    """Prepare input for the prediction query execution.

    There is two 'types' of input:
    1) split data: comes from splits. Multiple splits are combined using cross-joins.
    2) partial data: given by user. It's data of (a subspace of) the data space. It is intended to be used item-wise
       as input. Hence, its columns are not combined by cross join but used as is.

    Args:
        model: mb_modelbase.Model
        splits: [SplitTuple]
            The splits of the query.
        split_names: [str]
            The names of the split to generate input series for.
        partial_data: pd.DataFrame
            the partial input data.

    Returns: pd.DataFrame, dict<str:pd.Series>
        A 2-tuple of `partial_data` and a dict of <input split dimension name : generated pd.Series of values>
    """

    # normalize splits with method 'data' to partial_data
    partial_data = data_splits_to_partial_data(model, splits, partial_data)

    # generate input series for each input name
    name2split = dict(zip(split_names, splits))
    data_dict = {name: generate_input_series_for_dim(model, name, split_names, name2split, partial_data) for name in split_names}

    return partial_data, data_dict


def generate_input_series_for_dim(model, input_dim_name, split_names, name2split, partial_data):
    """Generate a pd.Series of input values for `input_dim_name`.

    Args:
        model: mb_modelbase.Model
        input_dim_name: str
            The name of the dimension to generate input for
        split_names: [str]
            The names of the split to generate input series for.
        name2split: dict<str, SplitTuple>
            dict of <name of dimension split> to <SplitTuple>
        partial_data: pd.DataFrame
            partial input data

    Returns: pd.Series
    """
    name = input_dim_name
    split_names = set(split_names)

    # if partial_data is available: use partial_data
    if name in partial_data.columns:
        series = partial_data.loc[:, name]

        # if split is available: modify partial_data
        if name in split_names:
            raise NotImplementedError('to be implemented. Currently overlapping partial data and splits are not '
                                      'possible')

    # if no partial_data is available but split: use split
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


def divide_df(df, colnames):
    """Splits df by colnames.

    Returns: (pd.DataFrame, pd.DataFrame)
        A tuple of two pd.DataFrames where the first one contains all columns with names in `colnames` and the second
        all other.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame

    colnames = set(colnames)
    df_colnames = set(df.columns)

    names_intersect = colnames.intersection(df_colnames)
    names_other = df_colnames - colnames

    assert(names_intersect.isdisjoint(names_other))

    return df[list(names_intersect)], df[list(names_other)]


def condition_ops_and_names(cond_names, name2split, split_dims, partial_dims):
    """Build operator list for `cond_names`.

    Builds a list of operators (string descriptors) for use in ConditionTuples for the names in `cond_names`.

    Args:
        name2split: dict
            Dictionary of split names to meta information.
        cond_names: list
            List of names of dimensions to to condition out. All names that is split by (i.e. names that are in name2split)
            occur first.
        split_dims: int
            number of split dimensions to condition on. Will be filled with operator from name2split.
        partial_dims: int
            number of partial data dimensions to condition on. Will be filled with operator "==".

    Returns: list(str)
        List of operators.
    """
    return [name2split[name]['cond_op'] for name in cond_names[:split_dims]] + ['=='] * partial_dims


def validate_input_data_for_density_probability(aggr, split_dfs, partial_df, name2split):
    """Validates and if possible 'corrects' input data for use with a density or probability aggregation.

    Density and probability aggregations require scalar- and domain-valued input, respectively. This method validates
    this. If it is violated it automatically corrects it using up-cast (scalar to domain) or down-cast (domain to
    scalar) functions. The casting methods are given as part of in `name2split`.

    TODO: currently the partial_data is assumed to be given at points only, not domains. Overcome this limitation and
     allow domains for partial_data as well.

    Args:
        aggr: AggregationTuple
        split_dfs: list(pd.Series)
        partial_df: pd.DataFrame
        name2split: dict<str, dict>
            Map of a name of a split to the split with meta information. See `make_name2split()`.
            
    Returns: list(pd.Series), pd.DataFrame 
        Possibly modified split_df_input and partial_df

    """

    method = aggr[METHOD_IDX]
    if method not in ['probability', 'density']:
        raise ValueError('up/down casting of input only makes sense for density/probability aggregations')

    def density_map_fct(series):
        # input data must all be scalar valued, i.e. splits must all have result type 'scalar'
        split = name2split[series.name]
        if split['return_type'] == 'domain':
            # down-cast from domains to scalar for density queries
            return series.apply(split['down_cast_fct'])
        else:
            return series

    def probability_map_fct(series):
        # input data must all be domain valued, i.e. splits must all have results type 'scalar'
        split = name2split[series.name]
        if split['return_type'] == 'scalar':
            # up-cast from scalar to domain for probability queries
            return series.apply(split['up_cast_fct'])
        else:
            return series

    if aggr[METHOD_IDX] == 'density':
        split_dfs = map(density_map_fct, split_dfs)
        # check partial input
        # TODO
    else:
        # check split input
        split_dfs = map(probability_map_fct, split_dfs)
        # check partial input
        # TODO

    return split_dfs, partial_df


def aggregate_density_or_probability(model, aggr, partial_df, split_series_dict, name2split, aggr_id='aggr_id'):
    """Compute density or probability aggregation `aggr` for `model` on given data.
    
    Args:
        model: mb_modelbase.Model
        aggr: AggregationTuple
        partial_df: pd.DataFrame
        split_series_dict: dict<str,pd.Series>
        name2split: dict<str, dict>
            Map of a name of a split to the split with meta information. See `make_name2split()`.
        aggr_id: str, optional, defaults to 'aggr_id'.
            name of column of the aggregation result in the resulting data frame.
    
    Returns: pd.DataFrame
        The result of the aggregation as a pd.DataFrame with input and output included and correctly named columns
    """

    # data has two 'types' of dimension:
    #  * `input_names`: input to the density query
    #  * `cond_out_names`: to be conditioned out
    input_names = model.sorted_names(aggr[NAME_IDX])
    cond_out_names = (set(partial_df.columns) | set(split_series_dict.keys())) - set(input_names)

    # divide partial_data and split data into those for conditioning out and for querying
    partial_df_cond_out, partial_df_input = divide_df(partial_df, cond_out_names)
    split_df_cond_out, split_df_input = [], []
    for name, df in split_series_dict.items():
        if name in cond_out_names:
            split_df_cond_out.append(df)
        else:
            split_df_input.append(df)

    # build cond out data and op list
    cond_out_data = crossjoin(*split_df_cond_out, partial_df_cond_out)
    cond_out_names = model.sorted_names(cond_out_names)
    cond_out_data = cond_out_data[cond_out_names]
    cond_out_ops = condition_ops_and_names(cond_out_names, name2split, len(split_df_cond_out),
                                           len(partial_df_cond_out.columns))

    # compute original, uncorrected input data. We will return this instead of the corrected input (below).
    # Doing differently causes inconsistency:
    #  - we might try to downcast/upcast later which would fail because it's been done already
    #  - merging with other results based on these inputs might fail because not all have been down/up casted the same
    input_data_orig = crossjoin(*split_df_input, partial_df_input)
    input_data_orig = input_data_orig[input_names]

    # TODO:
    #  model.predict(['sex', 'RW', Density([sex])], for_data=pd.DataFrame(data={'RW': [5, 10, 15], 'sex':['Male', 'Male', 'Female']}))
    #  this returns:
    #       sex  RW     density(['sex'])
    #      Male   5    0.495187393205537
    #      Male   5    0.495187393205537
    #    Female   5    0.504812606794463
    #      Male  10   0.6659190887603108
    #      Male  10   0.6659190887603108
    #    Female  10   0.3340809112396892
    #      Male  15  0.40305780181766876
    #      Male  15  0.40305780181766876
    #    Female  15   0.5969421981823313
    #   which is not what we want.
    #   The problem is that a cartesian product of the two partial data columns is taken because one is 'input' to
    #   the density query and the other is conditioned on. I didn't take this twist into consideration yet. Fix it!

    # validate and build input data
    split_df_input, partial_df_input = validate_input_data_for_density_probability(aggr, split_df_input,
                                                                                   partial_df_input, name2split)
    input_data = crossjoin(*split_df_input, partial_df_input)
    input_data = input_data[input_names]  # reorder to match model ordering

    # TODO: make the outer loop parallel
    method = aggr[METHOD_IDX]
    results = []
    if cond_out_data.empty:
        results = aggr_density_probability_inner(model, method, input_data)
    else:
        _extend = results.extend
        for row in cond_out_data.itertuples(index=False, name=None):
            # derive model for these specific conditions
            pairs = zip(cond_out_names, cond_out_ops, row)
            cond_out_model = model.copy().condition(pairs).marginalize(keep=input_names)
            # query model
            _extend(aggr_density_probability_inner(cond_out_model, method, input_data))

    # TODO: use multi indexes. this should give some speed up
    #  problem right now is: the columns for the keys are not hashable, because they contains lists
    #  solution: turn lists into tuples. Not sure how deep required changes would be
    # foo = crossjoin(cond_out_data, input_data)
    # df_index = pd.MultiIndex.from_frame(foo)
    # return_df = pd.DataFrame(data={aggr_id: results}, index=df_index)
    # return return_df

    # return full data frame of input and output
    return crossjoin(cond_out_data, input_data_orig).assign(**{aggr_id: results})


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


def aggregate_maximum_or_average(model, aggr, partial_data, split_series_dict, name2split, aggr_id='aggr_id'):
    """Compute maximum or average aggregation `aggr` for `model` on given data.

        Args:
            model: mb_modelbase.Model
            aggr: AggregationTuple
            partial_df: pd.DataFrame
            split_series_dict: dict<str,pd.Series>
            name2split: dict<str, dict>
                Map of a name of a split to the split with meta information. See `make_name2split()`.
            aggr_id: str, optional, defaults to 'aggr_id'.
                name of column of the aggregation result in the resulting data frame.

        Returns: pd.DataFrame
            The result of the aggregation as a pd.DataFrame with input and output included and correctly named columns
    """
    split_data_list = (df for name, df in split_series_dict.items())
    cond_out_data = crossjoin(*split_data_list, partial_data)
    cond_out_names = cond_out_data.columns
    cond_out_ops = condition_ops_and_names(cond_out_names, name2split, len(split_series_dict), len(partial_data.columns))

    # TODO: make the outer loop parallel
    # TODO: speed up results = np.empty(len(input_frame))
    results = []

    if len(cond_out_data) == 0:
        # there is no fields to split by, hence only a single value will be aggregated
        assert len(aggr[NAME_IDX]) == len(model.fields)
        res = model.aggregate(aggr[METHOD_IDX], opts=aggr[ARGS_IDX + 1])
        i = model.asindex(aggr[YIELDS_IDX])  # reduce to requested field
        results.append(res[i])

    else:
        row_id_gen = utils.linear_id_generator(prefix="_row")
        rowmodel_name = model.name + next(row_id_gen)

        # TODO: I think this can be speed up: no need to recompute `i`, i is identical for all iteration of for loop

        def pred_max_func(row, cond_out_names=cond_out_names, operator_list=cond_out_ops,
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
