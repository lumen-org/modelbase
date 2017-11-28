# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

This adds debug functionality to the Model class.

How to use it:
  1. import this module
  2. that's it! Now you will get very verbose output on calls to Model.density, Model.aggregate and Model.model

How does it work:
  * in short it creates wrapper functions that print debug information and binds them to the original method names
"""

import logging

from mb_modelbase.models_core import models as md

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# safe original references. They are reused in the debugging versions of the functions below.
_original_model = md.Model.model
_original_copy = md.Model._defaultcopy
_original_condition = md.Model.condition
_original_marginalize = md.Model._marginalize
_original_aggregate = md.Model.aggregate
_original_density = md.Model.density

_fail_str = " FAILED! see below"


def model_debug(self, model='*', where=[], as_=None):
    if isinstance(where, zip):
        where = list(where)
    name_string = "" if as_ is None else (" as " + as_)
    logger.debug("model " + md.model_to_str(self) + name_string + " as follows: ... ")
    m = _original_model(self, model, where, as_)
    logger.debug(".. modelling successful.")
    return m


def copy_debug(self, name=None):
    copy_str = "copy " + md.model_to_str(self) + " as " + (self.name if name is None else name)
    logger.debug(copy_str)
    model_copy = _original_copy(self, name)
    return model_copy


def condition_debug(self, conditions=[], is_pure=False):
    if isinstance(conditions, zip):
        conditions = list(conditions)
    model_str = "condition " + md.model_to_str(self)
    where_str = " such that " + md.conditions_to_str(self, conditions)
    log_str = model_str + where_str
    try:
        m = _original_condition(self, conditions, is_pure)
    except:
        logger.error(log_str + _fail_str)
        raise
    logger.debug(log_str + " ==> " + md.model_to_str(m))
    return m


def marginalize_debug(self, keep=None, remove=None):
    model_str = "marginalize " + md.model_to_str(self)
    if remove is None:
        remove = self.inverse_names(keep, sorted_=True)
    removing_str = " <nothing> " if len(remove) == 0 else " by removing " + md.name_to_str(self, remove)
    log_str = model_str + removing_str

    try:
        m = _original_marginalize(self, keep, remove)
    except:
        logger.error(log_str + _fail_str)
        raise
    logger.debug(log_str + " ==> " + md.model_to_str(m))
    return m


def aggregate_debug(self, method, opts=None):
    log_str = "arg-" + str(method) + "(" + md.model_to_str(self) + ")"
    try:
        aggr = _original_aggregate(self, method, opts)
    except:
        logger.error(log_str + _fail_str)
        raise
    logger.debug(log_str + " = " + str(aggr))
    return aggr


def density_debug(self, names, values=None):
    if values is None:
        # in that case the only argument holds the (correctly sorted) values
        name_strings = [md.field_to_str(field) for field in self.fields]
        values_ = names
    else:
        # name value pairs are given
        name_strings = [md.field_to_str(self.byname(name)) for name in names]
        values_ = values
    name_value_pairs = [name + "=" + str(val) for name, val in zip(name_strings, values_)]
    log_str = self.name + "(" + ",".join(name_value_pairs) + ")"
    try:
        p = _original_density(self, names, values)
        # p = self.density(names, values)
    except:
        logger.error(log_str + _fail_str)
        raise
    logger.debug(log_str + " = " + str(p))
    return p


# override original references
md.Model.model = model_debug
md.Model.aggregate = aggregate_debug
md.Model.density = density_debug
md.Model.condition = condition_debug
md.Model._marginalize = marginalize_debug  # override internal version!
md.Model._defaultcopy = copy_debug
