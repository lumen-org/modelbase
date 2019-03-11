# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas

This module contains helper functions methods to do prediction on models.


"""
import logging
import pandas as pd

from mb_modelbase.models_core import splitter as sp
from mb_modelbase.models_core.base import Split, Condition, Density
from mb_modelbase.models_core.base import NAME_IDX, METHOD_IDX, YIELDS_IDX, ARGS_IDX, OP_IDX, VALUE_IDX


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _tuple2str(tuple_):
    """Returns a string that summarizes the given splittuple or aggregation tuple"""
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


def add_split_for_defaulting_field(dim, splitby, split_ids, split_name2id, split_names, where, filter_names, idgen):
    """ A add a filter and identity split for the defaulting field `dim`, if possible.

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
    id_ = name + next(idgen)
    split_ids.append(id_)
    split_name2id[name] = id_
    split_names.append(name)

    # add condition
    condition = Condition(name, '==', def_)
    where.append(condition)
    filter_names.append(name)

    return def_


def apply_defaulting_fields(model, basenames,
                            aggrs, aggr_ids,
                            splitby, split_ids, split_names, split_name2id,
                            where, filter_names,
                            predict, predict_names, predict_ids,
                            evidence_names,
                            idgen):

    for t in predict:
        clause_type = type_of_clause(t)
        if clause_type == 'split':
            # t is a string, i.e. name of a field that is split by
            name = t
            predict_names.append(name)
            try:
                predict_ids.append(split_name2id[name])
            except KeyError:
                dim = model.byname(name)
                add_split_for_defaulting_field(dim, splitby, split_ids, split_name2id, split_names, where, filter_names, idgen)
                predict_ids.append(split_name2id[name])  # "retry"
            basenames.add(name)
        else:
            # t is an aggregation/density tuple
            id_ = _tuple2str(t) + next(idgen)
            aggrs.append(t)
            aggr_ids.append(id_)
            predict_names.append(_tuple2str(t))  # generate column name to return
            predict_ids.append(id_)
            aggr_input_dim_names = t[NAME_IDX]
            basenames.update(aggr_input_dim_names)

            if clause_type == 'density' or clause_type == 'probability':
                # all dimensions that are required as input by a density/probability must be available somehow:
                #  * as a split (i.e. splitby), or
                #  * from data (i.e. evidence)
                # if that is not the case: add a default split for it
                # TODO: this is not tested
                for name in aggr_input_dim_names:
                    if name not in evidence_names and name not in split_names:
                        add_split_for_defaulting_field(model.byname(name), splitby, split_ids, split_name2id, split_names,
                                                       where, filter_names, idgen)
                # TODO: just an idea: couldn't I merge the evidence given (takes higher priority) with all default of all variables and use this as a starting point for the input frame??


def derive_aggregation_model(basemodel, aggr, splitnames_unique, id_gen):
    """Derive a model from basemodel for the aggregation `aggr` considering that we will split along dimensions in
    split unique.

    Returns: mb_modelbase.Model
    """
    aggr_model = basemodel.copy(name=next(id_gen))
    if type_of_clause(aggr) == 'density':
        return aggr_model.model(model=aggr[NAME_IDX])
    else:
        return aggr_model.model(model=list(splitnames_unique | set(aggr[NAME_IDX])))


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


def generate_input_series(model, input_names_required, splits, split_names, evidence):
    """Returns a dict of pd.Series with an entry for every 'input dimension'

    Args:
        model: md_modelbase.Model
        input_names_required: sequence of str
            The names of the dimensions to generate input series' for.

    Returns: dict of <str:pd.Series>
        The generated dict of <input dimension name : series of input values>
    """

    name2split = dict(zip(split_names, splits))
    return {name:generate_input_series_for_dim(model, name, name2split, evidence) for name in input_names_required}


def generate_input_series_for_dim(model, input_dim_name, name2split, evidence):
    name = input_dim_name
    split_names = name2split.keys()

    # if evidence is available: use evidence
    if name in evidence.colnames:
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

