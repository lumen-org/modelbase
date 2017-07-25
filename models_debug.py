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
import models as md

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


# safe original references. They are reused in the debugging versions of the functions below.
_original_model = md.Model.model
_original_aggregate = md.Model.aggregate
_original_density = md.Model.density


def model_debug(self, model='*', where=[], as_=None):
    model_str = "model " + md.model_to_str(self)
    remove_names = self.inverse_names(self.names if model == "*" else model)
    removing_str = "" if len(remove_names) == 0 else " by removing " + md.name_to_str(self, remove_names)
    where_str = "" if len(where) == 0 else " where " + md.conditions_to_str(self, where)
    log_str = model_str + removing_str + where_str

    try:
        m = _original_model(self, model, where, as_)
    except:
        logger.error(log_str + "failed!!")
        raise
    logger.debug(log_str + " ==> " + md.model_to_str(m))
    return m


def aggregate_debug(self, method, opts=None):
    log_str = "arg-" + str(method) + "(" + md.model_to_str(self) + ")"
    try:
        aggr = _original_aggregate(self, method, opts)
    except:
        logger.error(log_str + "failed!!")
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
        logger.error(log_str, "failed!!")
        raise
    logger.debug(log_str + " = " + str(p))
    return p


# override original references
md.Model.model = model_debug
md.Model.aggregate = aggregate_debug
md.Model.density = density_debug
# md.Model.model_debug = model_debug
# md.Model.aggregate_debug = aggregate_debug
# md.Model.density_debug = density_debug
