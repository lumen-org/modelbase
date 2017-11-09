# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

This module defines:

   * Model: an abstract base class for models.
   * Field: a class that represent random variables in a model.
   * ConditionTuple, SplitTuple, AggregationTuple: convenience tuples for handling such clauses in PQL queries
"""
import copy as cp
from collections import namedtuple
from functools import reduce
import pickle as pickle
import numpy as np
import pandas as pd
import logging

import domains as dm
import splitter as sp
import utils as utils
import data_aggregation as data_aggr

logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)

""" Development Notes (Philipp)
# interesting links
https://github.com/rasbt/pattern_classification/blob/master/resources/python_data_libraries.md !!!

# what about using a different syntax for the model, like the following:

    model['A'] : to select a submodel only on random variable with name 'A' // marginalize
    model['B'] = ...some domain...   // condition
    model.
"""

AggregationTuple = namedtuple('AggregationTuple', ['name', 'method', 'yields', 'args'])
SplitTuple = namedtuple('SplitTuple', ['name', 'method', 'args'])
NAME_IDX = 0
METHOD_IDX = 1
YIELDS_IDX = 2
ARGS_IDX = 2
ConditionTuple = namedtuple('ConditionTuple', ['name', 'operator', 'value'])
OP_IDX = 1
VALUE_IDX = 2

""" A condition tuple describes the details of how a field of model is
    conditioned.

    Attributes:
        name: the name of field to condition
        operator: may take be one of ['in', 'equals', '==', 'greater', 'less']
        value: its allowed values depend on the value of operator
            operator is one of: 'in': A single element (to set the domain singular) or a sequence
                of elements (if the field is discrete), or a two-element list [min, max] if the
                field is continuous.
            operator is 'equals' or '==': a single element
            operator == 'greater': a single element that is set to be the new upper bound of the domain.
            operator == 'less': a single element that is set to be the new lower bound of the domain.
"""


def Field(name, domain, extent, dtype='numerical'):
    if not extent.isbounded():
        raise ValueError("extents must not be unbounded")
    if dtype not in ['numerical','string']:
        raise ValueError("dtype must be 'string' or 'numerical'")
    return {'name': name, 'domain': domain, 'extent': extent, 'dtype': dtype}
""" A constructor that returns 'Field'-dicts, i.e. a dict with three components
    as passed in:
        'name': the name of the field
        'domain': the domain of the field,
        'extent': the extent of the field that will be used as a fallback for domain if domain is unbounded but a
            value for domain is required
        'dtype': the data type that the field represents. Possible values are: 'numerical' and 'string'
"""


def _Modeldata_field():
    """Returns a new field that represents the imaginary dimension of 'model vs data'."""
    return Field('model vs data', dm.DiscreteDomain(), dm.DiscreteDomain(['model', 'data']), dtype='string')


""" Utility functions for converting models and parts / components of models to strings. """

def field_tojson(field):
    """Returns an adapted version of a field that in any case is JSON serializable. A fields domain may contain the
    value Infinity, which JSON serialized don't handle correctly.
    """
    # copy everything, but special treat domain and extent
    copy = cp.copy(field)
    copy['domain'] = field['domain'].tojson()
    copy['extent'] = field['extent'].tojson()
    return copy


def name_to_str(model, names):
    """Given a single name or a list of names of random variables, returns
    a concise string representation of these. See field_to_str()
    """
    return field_to_str(model.byname(names))


def field_to_str(fields):
    """Given a single field or a list of fields, returns a concise string representation of these. As EBNF:

        (#|±)<name>[*]

    where
        # denotes a categorical field
        ± denotes a numerical field
        <name> is the fields name
        * denotes a field with bound domain (i.e. it is already conditioned on it)
    """
    def _field_to_str(field):
        return ('#' if field['dtype'] == 'string' else '±') + field['name'] \
               + ("*" if field['domain'].isbounded() else "")   # * marks fields with bounded domains

    if isinstance(fields, dict):
        return _field_to_str(fields)
    else:
        lst = [_field_to_str(field) for field in fields]
        return "(" + ",".join(lst) + ")"


def model_to_str(model):
    field_strs = [field_to_str(field) for field in model.fields]
    prefix = "" if model.mode == "both" else (model.mode + "@")
    return prefix + model.name + "(" + ",".join(field_strs) + ")"


def condition_to_str(model, condition):
    field_str = field_to_str(model.byname(condition[NAME_IDX]))
    op_str = condition[OP_IDX]
    value_str = str(condition[VALUE_IDX])
    return field_str + op_str + value_str


def conditions_to_str(model, conditions):
    cond_strs = [condition_to_str(model, condition) for condition in conditions]
    return "(" + ",".join(cond_strs) + ")"


def _tuple2str(tuple_):
    """Returns a string that summarizes the given splittuple or aggregation tuple"""
    is_aggr_tuple = len(tuple_) == 4 and not tuple_[METHOD_IDX] == 'density'
    prefix = (str(tuple_[YIELDS_IDX]) + '@') if is_aggr_tuple else ""
    return prefix + str(tuple_[METHOD_IDX]) + '(' + str(tuple_[NAME_IDX]) + ')'

""" Utility functions for data import. """


def normalize_dataframe(df, numericals):
    """Normalizes all columns in data frame df. It uses z-score normalization and applies it per column. Returns the normalization parameters and the normalized dataframe,  as a tuple of (df, means, sigma). It expects only numercial columns in given dataframe.

    Args:
        df: dataframe to normalize.
    Returns:
        (df, means, sigmas): the normalized data frame, and the mean and sigma as np.ndarray
    """
    df = df.copy()
    numdf = df.loc[:, numericals]

    (n, dg) = numdf.shape
    means = numdf.sum(axis=0) / n
    sigmas = np.sqrt((numdf ** 2).sum(axis=0) / n - means ** 2)

    df.loc[:, numericals] = (numdf - means) / sigmas

    return df, means.values, sigmas.values


def clean_dataframe(df):
    # check that there are no NaNs or Nones
    if df.isnull().any().any():
        raise ValueError("DataFrame contains NaNs or Nulls.")

    # convert any categorical columns that have numbers into strings
    # and raise errors for unsupported dtypes
    for colname in df.columns:
        col = df[colname]
        dtype = col.dtype
        if dtype.name == 'category':
            # categories must have string levels
            cat_dtype = col.cat.categories.dtype
            if cat_dtype != 'str' and cat_dtype != 'object':
                logger.warning('Column "' + str(colname) +
                               '" is categorical, however the categories levels are not of type "str" or "object" '
                               'but of type "' + str(cat_dtype) +
                               '". I\'m converting the column to dtype "object" (i.e. strings)!')
                df[colname] = col.astype(str)

    return df


def get_columns_by_dtype(df):
    """Returns a triple of colnames (all, cat, num) where:
      * all is all names of columns in df,
      * cat is the names of all categorical columns in df, and
      * num is the names of all numerical columns in df.
      Any column in df that is not recognized as either categorical or numerical will raise a TypeError.
      """
    all = []
    categoricals = []
    numericals = []
    for colname in df:
        column = df[colname]
        if column.dtype.name == "category" or column.dtype.name == "object":
            categoricals.append(colname)
        elif np.issubdtype(column.dtype, np.number):
            numericals.append(colname)
        else:
            raise TypeError("unsupported column dtype : " + str(column.dtype.name) + " of column " + str(colname))
        all.append(colname)
    return all, categoricals, numericals


def get_discrete_fields(df, colnames):
    """Returns discrete fields constructed from the columns in colname of dataframe df.
    This assumes colnames only contains names of discrete columns of df."""
    fields = []
    for colname in colnames:
        column = df[colname]
        domain = dm.DiscreteDomain()
        extent = dm.DiscreteDomain(sorted(column.unique()))
        field = Field(colname, domain, extent, 'string')
        fields.append(field)
    return fields


def get_numerical_fields(df, colnames):
    """Returns numerical fields constructed from the columns in colname of dataframe df.
    This assumes colnames only contains names of numerical columns of df."""
    fields = []
    for colname in colnames:
        column = df[colname]
        field = Field(colname, dm.NumericDomain(), dm.NumericDomain(column.min(), column.max()), 'numerical')
        fields.append(field)
    return fields


class Model:
    """An abstract base model that provides an interface to derive submodels from it or query density and other
    aggregations of it. It also defines stubs for those methods that actual models are required to implement. The stubs are:

      * _set_data(self, df, drop_silently)
      * _fit(self)
      * _marginalizeout(self, keep, remove)
      * _conditionout(self, keep, remove)
      * _density(self, x)
      * _sample(self)
      * copy(self, name=None)

    A model has a number of fields (aka dimensions). The model models a probability density function on these fields
    and allows various queries again this density. The model is based on data (aka evidence), and the data can be
    queried the same way like the model.

    In fact unified queries are possible, by means of an additional 'artificial' field 'model vs data'. That is an
    ordinal field with the two values "model" and "data". It can only be filtered or split by this field, and the
    results are accordingly for the results from the data or the model, respectively.

    Internal:
        Internally queries against the model and its data are answered differently. Also that field does not actually
        exists in the model. Queries may contain it, however they are translated onto the 'internal' interface in the
        .marginalize, .condition, .predict-methods.
    """

    def __str__(self):
        # TODO: add some more useful print out functions / info to that function
        return (self.__class__.__name__ + " " + self.name + "':\n" +
                "dimension: " + str(self.dim) + "\n" +
                "names: " + str([self.names]) + "\n")
                # "names: " + str([self.names]) + "\n" +
                # "fields: " + str([str(field) for field in self.fields]))

    def __short_str__(self, max_fields=5):
        fields = self.fields[:max_fields]
        field_strs = [('#' if field.dtype == 'string' else '±') + field.name for field in fields]
        return self.name + "(" + ",".join(field_strs) + ")"

    def asindex(self, names):
        """Given a single name or a list of names of random variables, returns
        the indexes of these in the .field attribute of the model.

        Note that this function cannot resolve the special field name 'model vs data', as that field has no internal index.
        """
        if isinstance(names, str):
            return self._name2idx[names]
        else:
            return [self._name2idx[name] for name in names]

    def byname(self, names):
        """Given a list of names of random variables, returns the corresponding
        fields of this model.

        Note that this function correctly resolve the special field name 'model vs data'.
        """
        def _byname(name):
            try:
                return self.fields[self._name2idx[name]]
            except KeyError as ke:
                if ke.args[0] == 'model vs data':
                    return self._modeldata_field
                raise

        if isinstance(names, str):
            return _byname(names)
        else:
            return [_byname(name) for name in names]

    def isfieldname(self, names):
        """Returns true iff the single string or list of strings given as variables names are (all) names of random
        variables of this model.

        Note that this function correctly handles the special field name 'model vs data'.
        """
        if isinstance(names, str):
            return names in self._name2idx
        else:
            # TODO try:
            # return all([name in self._name2idx or name == 'model vs data' for name in names])
            return all([name in self._name2idx for name in names])

    def inverse_names(self, names, sorted_=False):
        """Given a sequence of names of random variables (or a single name), returns a sorted list of all names
        of random variables in this model which are _not_ in names. The order of names is the same as the order of
        fields in the model.
        It ignores silently any name that is not a name of a field in this model.

        Args:
            names: a sequence of names (strings)
            sorted_: Boolean flag that indicates whether or not names is in sorted order with respect to the
             order of fields in this model.
        """
        if isinstance(names, str):
            names = [names]
            sorted_ = True
        if sorted_:
            return utils.invert_sequence(names, self.names)
        else:
            names = set(names)
            return [name for name in self.names if name not in names]

    def sorted_names(self, names):
        """Given a set, sequence or list of random variables of this model, returns a list of the
        same names but in the same order as the random variables in the model.
        old: It silently ignores any name that is not a name of a field in this model.
        """
        assert(len(names) <= len(self.names))
        return utils.sort_filter_list(names, self.names)

    def __init__(self, name):
        self.name = name
        # the following is all done in _fields_set_empty, which is called below
        # self.fields = []
        # self.names = []
        # self.extents = []
        # self.dim = 0
        # self._name2idx = {}
        self._fields_set_empty()
        self.data = pd.DataFrame([])
        self._aggrMethods = None
        self.mode = "empty"
        self._modeldata_field = _Modeldata_field()


    def _setempty(self):
        self._update_remove_fields()
        return self

    def _isempty(self):
        return self.dim == 0

    def json_fields(self, include_modeldata_field=False):
        """Returns a json-like representation of the fields of this model."""
        json_ = list(map(field_tojson, self.fields))
        if include_modeldata_field:
            json_.append(field_tojson(self._modeldata_field))
        return json_

    def set_data(self, df, silently_drop=False):
        """Derives suitable, cleansed (and copied) data from df for a particular model, sets this as the models data
         and also sets up auxiliary data structures if necessary. This is however, only concerned with the data
         part of the model and does not do any fitting of the model.

        This method has the following effects:
         - the models mode is set to 'data'
         - the models data is set to some model-specific cleaned version of the data.
         - the models fields attribute is derived accordingly
         - possibly existing data of the model are overwritten
         - possibly fitted model parameters are lost

        Note that if the data does not fit to the specific type of the model, it will raise a TypeError. E.g. a gaussian
        model cannot be fit on categorical data.

        Args:
            df: a pandas data frame
            silently_drop: If set to True any column of df that is not suitable for the model to be learned will silently be dropped. Otherwise this will raise a TypeError.

        Returns:
            self
        """
        # general clean up
        df = clean_dataframe(df)

        # model specific clean up, setting of data, models fields, and possible more model specific stuff
        callbacks = self._set_data(df, silently_drop)
        self.mode = 'data'
        self._update_all_field_derivatives()
        for callback in callbacks:
            callback()

        return self

    def _set_data(self, df, silently_drop):
        """ This method sets the data for this model instance using the data frame df. After completion of this method
        the following attributes are set:
         * self.data
         * self.fields
         * self.mode
        See also model.set_data.

        This method must be implemented by all actual model classes. You might just want to use one of the following:
         * Model._set_data_mixed: for continuous and categorical data
         * Model._set_data_continuous: for continuous data
         * Model._set_data_categorical: for categorical data
        """
        raise NotImplementedError("your model must implement this method")

    def _set_data_mixed(self, df, silently_drop):
        """see Model._set_data"""

        # split in categorical and numeric columns
        _, self._categoricals, self._numericals = get_columns_by_dtype(df)

        # check if dtype are ok
        #  ... nothing to do here ...

        # set data for model as needed
        #  reorder data frame such that categorical columns are first
        df = df[self._categoricals + self._numericals]

        # shuffle data frame
        df = df.sample(frac=1).reset_index(drop=True)

        # derive and set fields
        self.fields = get_discrete_fields(df, self._categoricals) + \
                 get_numerical_fields(df, self._numericals)

        self.data = df

        # normalize numerical part of data frame. For better numerical performance.
        # this is done at model level
        #if normalize:
        #        ) = normalize_dataframe(df.loc[:, self._numericals])
        #        self.data_norm =

        return self

    def _set_data_continuous(self, df, silently_drop):
        """see Model._set_data"""

        # split in categorical and numeric columns
        all, categoricals, numericals = get_columns_by_dtype(df)

        # check if dtype are ok
        if len(categoricals) > 0:
            raise ValueError('Data frame contains categorical columns: ' + str(categoricals))

        # set data for model as needed
        self.data = pd.DataFrame(df, columns=numericals)

        # derive and set fields
        self.fields = get_numerical_fields(self.data, numericals)

        return self

    def _set_data_categorical(self, df, silently_drop):
        """see Model._set_data"""

        # split in categorical and numeric columns
        all, categoricals, numericals = get_columns_by_dtype(df)

        # check if dtype are ok
        if len(numericals) > 0:
            raise ValueError('Data frame contains numerical columns: ' + str(numericals))

        # set data for model as needed
        self.data = pd.DataFrame(df, columns=categoricals)

        # derive and set fields
        self.fields = get_discrete_fields(self.data, categoricals)
        return self

    def fit(self, df=None, **kwargs):
        """Fits the model to a models data or an optionally passed DataFrame

        Note that on return of this method the attribute .data must be filled with the appropriate data that
        was used to fit the model.

        Args:
            df: Optional. A pandas data frame that holds the data to fit the model to. You can also previously set the
             data to fit to using the set_data method.

        Returns:
            The fitted model.
        """

        if df is not None:
            return self.set_data(df).fit(**kwargs)

        if df is None and self.mode != 'data':
            raise ValueError('No data frame to fit to present: pass it as an argument or set it before using set_data(df)')

        try:
            callbacks = self._fit(**kwargs)
            for callback in callbacks:
                callback()
        except NameError:
            raise NotImplementedError("You have to implement the _fit method in your model!")
        self.mode = "both"
        return self

    def marginalize(self, keep=None, remove=None, is_pure=False):
        """Marginalizes random variables out of the model. Either specify which
        random variables to keep or specify which to remove.

        Note that marginalization depends on the domain of a random variable. That is: if
        its domain is bounded it is conditioned on this value (and marginalized out).
        Otherwise it is 'normally' marginalized out (assuming that the full domain is available).

        Arguments:
            keep: A list of names of random variables of this model to keep. All other random variables
                are marginalized out.
            remove: A list of names of random variables  of this model to marginalize out.
            is_pure: An performance optimization parameter. Set to True if you can guarantee that keep and remove do not contain the special value "model vs data".

        Returns:
            The modified model.
        """
        #logger.debug('marginalizing: ' + ('keep = ' + str(keep) if remove is None else ', remove = ' + str(remove)))

        if keep is not None and remove is not None:
            raise ValueError("You may only specify either 'keep' or 'remove', but non both.")
        if keep is not None:
            if keep == '*':
                keep = self.names
            elif not is_pure:
                keep = [name for name in keep if name != 'model vs data']
            if not self.isfieldname(keep):
                raise ValueError("invalid random variables names in argument 'keep': " + str(keep))
        elif remove is not None:
            if not is_pure:
                remove = [name for name in remove if name != 'model vs data']
            if not self.isfieldname(remove):
                raise ValueError("invalid random variable names in argument 'remove': " + str(remove))
            keep = set(self.names) - set(remove)
        else:
            raise ValueError("Missing required argument: You must specify either 'keep' or 'remove'.")

        if len(keep) == self.dim or self._isempty():
            return self
        if len(keep) == 0:
            return self._setempty()

        keep = self.sorted_names(keep)
        return self._marginalize(keep)

    def _marginalize(self, keep=None, remove=None):
        """Marginalize the fields given in remove, keeping those in keep.

        This is the internal version of the public method marginalize. You probably want to call that one.

        Args:
            keep: the names of the fields to keep, in the same order than in self.fields.
            remove: the names of the fields to remove, in the same order than in self.fields.

        Notes:
            You have to specify at least one of keep and remove.
            Neither keep nor remove may contain the field 'model vs data'.
        """
        if keep is not None and remove is not None:
            assert(set(keep) | set(remove) == set(self.names))
        else:
            if remove is None:
                remove = self.inverse_names(keep, sorted_=True)
            if keep is None:
                keep = self.inverse_names(remove, sorted_=True)
        # else:
        #     raise ValueError("specify at least one of 'keep' and 'remove'!")
        cond_out = [name for name in remove if self.byname(name)['domain'].isbounded()]

        # Data marginalization
        if self.mode == 'both' or self.mode == 'data':
            self.data = self.data.loc[:, keep]

        # Model marginalization
        # there are three cases of marginalization:
        # (1) unrestricted domain, (2) restricted, but not singular domain, (3) singular domain
        # we handle case (2) and (3) in ._conditionout, then case (1) in ._marginalizeout
        if len(cond_out) != 0:
            callbacks = self._conditionout(self.inverse_names(cond_out, sorted_=True), cond_out)
            self._update_remove_fields(cond_out)
            for callback in callbacks:
                callback()
            remove = self.inverse_names(keep, sorted_=True)  # do it again, because fields may have changed ??

        if len(keep) == self.dim or self._isempty():
            return self

        callbacks = self._marginalizeout(keep, remove)
        self._update_remove_fields(remove)
        for callback in callbacks:
            callback()
        return self

    def _marginalizeout(self, keep, remove):
        """Marginalizes the model such that only random variables with names in keep remain and
        variables with name in remove are removed.

        This method must be implemented by any actual model that derived from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
          * keep is anything but a list of names of random variables of this model
          * keep is empty
          * the model itself is empty
          * keep contains all names of the model
          * remove is anything but the inverse list of names of fields with respect to keep

        Moreover keep and remove are guaranteed to be in the same order than the random variables of this model
        """
        raise NotImplementedError("Implement this method in your model!")

    def _condition_data(self, name, operator, values):
        """Conditions the data of the model according to given parameters. Returns nothing.
        """
        df = self.data  # shortcut
        field = self.byname(name)
        column = df[name]
        if operator == 'in':
            if field['dtype'] == 'numerical':
                df = df.loc[column.between(*values, inclusive=True)]
            elif field['dtype'] == 'string':
                df = df.loc[column.isin(values)]
            else:
                raise TypeError("unsupported field type: " + str(field.dtype))
        else:
            # values is necessarily a single scalar value, not a list
            if operator == 'equals' or operator == '==':
                df = df.loc[column == values]
            elif operator == 'greater' or operator == '>':
                df = df.loc[column > values]
            elif operator == 'less' or operator == '<':
                df = df.loc[column < values]
            else:
                raise ValueError('invalid operator for condition: ' + str(operator))
        self.data = df

    def condition(self, conditions=[], is_pure=False):
        """Conditions this model according to the list of three-tuples
        (<name-of-random-variable>, <operator>, <value(s)>). In particular
        objects of type ConditionTuples are accepted and see there for allowed values.

        Note: This only restricts the domains of the random variables. To
        remove the conditioned random variable you need to call marginalize
        with the appropriate parameters.

        Returns:
            The modified model.
        """

        # TODO: simplify the interface?

        # can't do that because I want to allow a zip object as conditions...
        # if len(conditions) == 0:
        #     return self

        # treat model vs data field correctly. It must be applied first.
        names = []
        if is_pure:
            pure_conditions = conditions
        else:
            # allow conditioning on modeldata_field
            # find 'model vs data' field and apply conditions
            pure_conditions = []
            for condition in conditions:
                (name, operator, values) = condition
                if name == 'model vs data':
                    domain = self._modeldata_field['domain']
                    domain.apply(operator, values)
                    # REMOVE SOON:
                    # if operator == 'in' or operator == 'equals' or operator == '==':
                    #     domain.intersect(values)
                    # else:
                    #     raise ValueError('invalid operator for condition: ' + str(operator))
                    # set internal mode accordingly to both, data or model
                    value = domain.value()
                    self.mode = value if value == 'model' or value == 'data' else 'both'
                else:
                    names.append(name)
                    pure_conditions.append(condition)

        # continue with rest according to mode
        if self.mode == 'model':
            for (name, operator, values) in pure_conditions:
                self.byname(name)['domain'].apply(operator, values)
        else:
            for (name, operator, values) in pure_conditions:
                # condition model. actually, this only restricts the domain of the fields
                self.byname(name)['domain'].apply(operator, values)
                # condition data
                self._condition_data(name, operator, values)
        self._update_extents(names)
        return self

    def _conditionout(self, keep, remove):
        """Conditions the random variables with name in remove on their available, //not unbounded// domain and
        marginalizes them out.

        This method must be implemented by any actual model that derived from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
          * remove is anything but a list of names of random-variables of this model
          * remove is empty
          * the model itself is empty
          * remove contains all names of the model
          * keep is anything but the inverse list of names of fields with respect to remove

        Moreover keep and remove are guaranteed to be in the same order than the random variables of this model

        Note that we don't know yet how to condition on a non-singular domain (i.e. condition on interval or sets).
        As a work around we therefore:
          * for continuous domains: condition on (high-low)/2
          * for discrete domains: condition on the first element in the domain
        """
        raise NotImplementedError("Implement this method in your model!")

    def aggregate_data(self, method, opts=None):
        """Aggregate the models data according to given method and options, and returns this aggregation."""
        data = self.data
        if len(data) == 0:
            # can not compute any aggregation. return nan
            # TODO: allows Nans
            # raise ValueError("empty data frame - cannot compute any aggregations. implement nans.")
            return [0] * self.dim
        elif method == 'maximum' or method == 'average':
            # return data_aggr.most_frequent_equi_sized(data, opts)  # this is also an option, but I think it's worse
            return data_aggr.average_most_frequent(data, opts)
        else:
            raise ValueError("invalid value for method: " + str(method))

    def aggregate_model(self, method, opts=None):
        """Aggregate the model according to given method and options and return the aggregation value."""

        # need index to merge results later
        other_idx = []
        other_names = []
        singular_idx = []
        singular_names = []
        singular_res = []

        # 1. find singular fields (fields with singular domain)
        # any aggregation on such fields must yield the singular value of the domain

        # need to copy model because we change the domain for the above heuristic... that seems ugly, but it's not too
        # bad because we would anyway need to copy the model before conditioning out singular fields
        model = self.copy()

        for (idx, field) in enumerate(model.fields):
            domain = field['domain']
            issingular = domain.issingular()
            isbounded = domain.isbounded()

            # Problem: how to marginalize if it is bounded but not singular?
            # Solution: apply heuristic to reduce any bounded domain to singular domain
            # see also: http://wiki.inf-i2.uni-jena.de/doku.php?id=emv:models:restrictions&#marginalization_of_interval-conditioned_fields
            if not issingular and isbounded:
                extent = field['extent']
                condition_scalar = domain.intersect(extent).mid()  # compute value to condition on
                domain.intersect(condition_scalar)  # condition, i.e. restrict domain to scalar
                issingular = True

            if issingular:
                singular_idx.append(idx)
                singular_names.append(field['name'])
                singular_res.append(domain.value())
            else:
                other_names.append(field['name'])
                other_idx.append(idx)

        # quit early if possible
        if len(other_idx) == 0:
            model_res = singular_res
        else:
            # 2. marginalize singular fields out
            # TODO: use internal, faster version of marginalize / marginalizeout?
            #submodel = self if len(singular_names) == 0 else self.copy()._marginalize(remove=singular_names)
            model = model if len(singular_names) == 0 else model._marginalize(keep=other_names, remove=singular_names)

            # 3. calculate 'unrestricted' aggregation on the remaining model
            try:
                aggr_function = model._aggrMethods[method]
            except KeyError:
                raise ValueError("Your model does not provide the requested aggregation: '" + method + "'")
            other_res = aggr_function()

            # 4. clamp to values within domain
            # TODO bug/mistake: should we really clamp?
            for (idx, field) in enumerate(model.fields):
                other_res[idx] = field['domain'].clamp(other_res[idx])

            # 5. merge with singular results
            model_res = utils.mergebyidx(singular_res, other_res, singular_idx, other_idx)
        return model_res

    def aggregate(self, method, opts=None):
        """Aggregates this model using the given method and returns the
        aggregation as a list. The order of elements in the list matches the
        order of random variables in fields attribute of the model.

        Returns:
            The aggregation of the model. It always returns a list, even if it contains only a single value.
        """
        if self._isempty():
            raise ValueError('Cannot aggregate a 0-dimensional model')
        # the data part can be aggregated without any model specific code and merged into the model results at the end
        # see my notes for how to calculate a single aggregation
        mode = self.mode
        if mode == "both":
            return self.aggregate_model(method, opts), self.aggregate_data(method, opts)
        if mode == "model":
            return self.aggregate_model(method, opts)
        if mode == "data":
            return self.aggregate_data(method, opts)
        raise ValueError("invalid value for mode : ", str(mode))

    def density(self, names, values=None):
        """Returns the density at given point. You may either pass both, names
        and values, or only one list with values. In the latter case values is
        assumed to be in the same order as the random variables of the model.

        Also note that you may pass the special field with name 'model vs data'.
        If the value is 'data' then it is interpreted as a density query against
        the data, i.e. frequency of items is returned. If the value is 'model'
        it is interpreted as a query against the model, i.e. density of input
        is returned. If value is 'both' a 2-tuple of both is returned.

        If you pass the 'model vs data' field, you must either also specify its name
        or you must specify it as the last value.

        Args:
            values: may be anything that numpy.array accepts to construct from.
        """

        # TODO: design: the way of handling the 'model vs data' field sucks ... how to do it better?
        # TODO: also, this is a lot of code for just querying density!? we need a _fast_ way to do that,
        # i.e. an interface that is used internally needs little preprocessing
        # TODO/idea: different way of passing it in: as a dict of name-of-field : value ?

        if self._isempty():
            raise ValueError('Cannot query density of 0-dimensional model')

        if values is not None and len(names) != len(values):
            raise ValueError('Length of names and values does not match.')

        mode = self.mode
        if len(names) != self.dim:
            # it may be that the 'model vs data' field was passed in
            if len(names) == self.dim+1:
                if values is None:
                    mode = names.pop()
                else:
                    names_ = []
                    values_ = []
                    # find 'model vs data' and rebuild accordingly shortened arguments
                    for (name, value) in zip(names, values):
                        if name == 'model vs data':
                            mode = value
                        else:
                            names_.append(name)
                            values_.append(value)
                    names = names_
                    values = values_
            if len(names) > self.dim:
                raise ValueError('Incorrect number names/values provided. Require ' + str(self.dim) +
                                 ' but got ' + str(len(names)) + '.')

        if values is None:
            # in that case the only argument holds the (correctly sorted) values
            values = names
        else:
            # names and values need sorting
            sorted_ = sorted(zip(self.asindex(names), values), key=lambda pair: pair[0])
            values = [pair[1] for pair in sorted_]

        # data frequency
        def filter_(df, conditions):
            """ Apply all '==' filters in the sequence of conditions to given dataframe and return it."""
            # TODO: do I need some general solution for the interval vs scalar problem?
            for (col_name, value) in conditions:
                try:
                    if not isinstance(value, str):  # try to access its elements
                        # TODO: this really is more difficult. in the future we want to support values like ['A', 'B']
                        # TODO: I guess I should pass a (list of) Domain to the density-method in the first place
                        # assuming interval for now
                        df = df.loc[df[col_name].between(*value, inclusive=True)]
                    else:
                        df = df.loc[df[col_name] == value]
                except TypeError:  # catch when expansion (*value) fails. its a scalar then...
                    df = df.loc[df[col_name] == value]
            return df

        def reduce_to_scalars(values):
            """Reduce all elements of values to scalars. Scalars will be kept. Intervals are reduced to its mean."""
            v = []
            for value in values:
                # all of the todos in filter apply here as well...
                try:
                    if not isinstance(value, str):
                        v.append((value[0]+value[1])/2)
                    else:
                        v.append(value)
                except (TypeError, IndexError):
                    v.append(value)
            return v

        # put debug / log code here
        # TODO ??

        if mode == "both" or mode == "data":
            cnt = len(filter_(self.data, zip(self.names, values)))
            if mode == "data":
                return cnt

        p = self._density(reduce_to_scalars(values))
        if mode == "model":
            return p
        elif mode == "both":
            return p, cnt
        else:
            raise ValueError("invalid value for mode : ", str(mode))

    def _density(self, x):
        """Returns the density of the model at point x.

        This method must be implemented by any actual model that derives from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
          * x is anything but a list of values as input for density
          * x has a length different than the dimension of the model
          * the model itself is empty

        It is _not_ guaranteed, however, that the elements of x are of matching type / value for the model

        Args:
            x: a list of values as input for the density.
        """
        raise NotImplementedError("Implement this method in your model!")

    def sample(self, n=1):
        """Returns n samples drawn from the model as a dataframe with suitable column names."""
        if self._isempty():
            raise ValueError('Cannot sample from 0-dimensional model')
        samples = (self._sample() for i in range(n))
        return pd.DataFrame.from_records(data=samples, columns=self.names)

    def _sample(self):
        """Returns a single sample drawn from the model.

        This method must be implemented by any actual model that derives from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
            * the model is empty
        """
        raise NotImplementedError()

    def copy(self, name=None):
        """Returns a copy of this model, optionally using a new name.

        This method must be implemented by any actual model that derives from the abstract Model class.

        Note: you must use Model._defaultcopy() to get a copy of the 'abstract part' of an model
        instance and then only add your custom copy code.
        """
        raise NotImplementedError()

    def _defaultcopy(self, name=None):
        """Returns a new model of the same type with all instance variables of the abstract base model copied:
          * data (a reference to it)
          * fields (deep copy)
        """
        name = self.name if name is None else name
        mycopy = self.__class__(name)
        mycopy.data = self.data
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy.mode = self.mode
        mycopy._modeldata_field = cp.deepcopy(self._modeldata_field)
        mycopy._update_all_field_derivatives()
        return mycopy

    def _condition_values(self, remove, pairflag=False):
        """Returns the list of values to condition on given a sequence of random variable names to condition on.
        Essentially, this is a look up in the domain restrictions of the fields to be conditioned on.

        Args:
            remove: sequence of random variable names
            pairflag = False: Optional. If set True not a list of values but a zip-object of the names and the
             values to condition on is returned
        """
        # TODO: we don't know yet how to condition on a not singular, but not unrestricted domain.
        cond_values = []
        for name in remove:
            field = self.byname(name)
            domain = field['domain']
            dvalue = domain.value()
            if not domain.isbounded():
                raise ValueError("cannot condition random variables with not bounded domain!")
            if field['dtype'] == 'numerical':
                cond_values.append(dvalue if domain.issingular() else (dvalue[1] + dvalue[0]) / 2)
            elif field['dtype'] == 'string':
                cond_values.append(dvalue if domain.issingular() else dvalue[0])
            else:
                raise ValueError('invalid dtype of field: ' + str(field['dtype']))

        return zip(remove, cond_values) if pairflag else cond_values

    @staticmethod
    def save(model, filename):
        with open(filename, 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as input:
            model = pickle.load(input)
            if not isinstance(model, Model):
                raise TypeError('pickled input is not an instance of Model.')
            return model

    def _fields_set_empty(self):
        self.fields = []
        self.names = []
        self.extents = []
        self.dim = 0
        self._name2idx = {}
        return

    def _update_extents(self, to_update=None):
        """Updates self.extents of the fields with names in the sequence to_update. Updates all if to_update is None.
        @fields
        """
        if to_update is None:
            to_remove = self.names
        to_update_idx = self.asindex(to_update)
        for idx in to_update_idx:
            field = self.fields[idx]
            self.extents[idx] = field['domain'].bounded(field['extent'])
        return self

    def _update_name2idx_dict(self):
        """Updates (i.e. recreates) the _name2idx dictionary from current self.fields."""
        self._name2idx = dict(zip([f['name'] for f in self.fields], range(self.dim)))

    def _update_remove_fields(self, to_remove=None):
        """Removes the fields in the sequence to_remove (and correspondingly derived structure) from self.fields.
         Removes all if to_remove is None
         @fields
        """
        if to_remove is None:
            return self._fields_set_empty()

        to_remove_idx = self.asindex(to_remove)
        # assert order of indexes
        assert(all(to_remove_idx[i] <= to_remove_idx[i+1] for i in range(len(to_remove_idx)-1)))

        # TODO: I guess it is much faster to recreate that list ...
        # however in the solution now we never change what object we reference to by self.name, ...
        for idx in reversed(to_remove_idx):
            del self.names[idx]
            del self.extents[idx]
            del self.fields[idx]

        self.dim = len(self.fields)
        self._update_name2idx_dict()
        return self

    def _update_all_field_derivatives(self):
        """Rebuild all field derivatives. self.fields must be set before calling
        @fields
        """
        self.dim = len(self.fields)
        self._update_name2idx_dict()
        self.names = [f['name'] for f in self.fields]
        self.extents = [field['domain'].bounded(field['extent']) for field in self.fields]
        return self

    def model(self, model='*', where=[], as_=None):
        """Returns a model with name 'as_' that models the random variables in 'model'
        respecting conditions in 'where'.

        Note that it does NOT create a copy, but modifies this model.

        Args:
            model:  A list of strings, representing the names of random variables to
                model. Its value may also be "*" or ["*"], meaning all random variables
                of this model.
            where: A list of 'conditiontuple's, representing the conditions to
                model.
            as_: An optional string. The name for the model to derive. If set
                to None the name of the base model is used. Defaults to None.

        Returns:
            The modified model.
        """
        self.name = self.name if as_ is None else as_
        return self.condition(where).marginalize(keep=model)

    def predict(self, predict, where=[], splitby=[], returnbasemodel=False):
        """Calculates the prediction against the model and returns its result
        by means of a data frame.

        The data frame contains exactly those columns/random variables which
        are specified in 'predict'. Its order is preserved.

        Note that this does NOT modify the model it is called on.

        Args:
            predict: A list of names of fields (strings) and 'AggregationTuple's.
                This is the list of fields to be included in the returned dataframe.
            where: A list of 'conditiontuple's, representing the conditions to
                adhere.
            splitby: A list of 'SplitTuple's, i.e. a list of fields on which to
                split the model and the method how to do the split.
            returnbasemodel: A boolean flag. If set this method will return the
                pair (result-dataframe, basemodel-for-the-prediction).
                Defaults to False.
        Returns:
            A dataframe with the fields as given in 'predict', or a tuple (see
            returnbasemodel).

        Internal:
            .predict accepts the 'artificial field' 'model vs data' ('mvd' for short in the following)
            in order to unify queries again model and data. It also translates it into to internal interfaces
            for model queries and data queries.
        """
        # TODO: add default splits for each data type?
        # TODO: improve interface: allow scalar arguments instead of lists too

        if self._isempty():
            return pd.DataFrame()
        #    return (pd.DataFrame(), pd.DataFrame()) if mode == 'both' else pd.DataFrame()
        idgen = utils.linear_id_generator()

        # How to handle the 'artificial field' 'model vs data':
        # The possible cases, as which 'model vs data' occur are:
        #   * as a filter: that sets the internal mode of the model. Is handled in '.condition'
        #   * as a split: splits by the values of that field. Is handled where we handle splits.
        #   * not at all: defaults to a filter to equal 'model' and a default identity split on it, ONLY if
        #       self.mode is 'both'
        #   * in predict-clause or not -> need to assemble result frame correctly
        #   * in aggregation: raise error

        filter_names = [f[NAME_IDX] for f in where]
        split_names = [f[NAME_IDX] for f in splitby]  # name of fields to split by. Same order as in split-by clause.

        # set default filter and split on 'model vs data' if necessary
        if 'model vs data' not in split_names \
                and 'model vs data' not in filter_names\
                and self.mode == 'both':
            where.append(('model vs data', '==', 'model'))
            filter_names.append('model vs data')
            splitby.append(('model vs data', 'identity', []))
            split_names.append('model vs data')

        # (1) derive base model, i.e. a model on all requested dimensions and measures, respecting filters
        predict_ids = []  # unique ids of columns in data frame. In correct order. For reordering of columns.
        predict_names = []  # names of columns as to be returned. In correct order. For renaming of columns.

        split_ids = [f[NAME_IDX] + next(idgen) for f in
                     splitby]  # ids for columns for fields to split by. Same order as in splitby-clause.
        split_name2id = dict(zip(split_names, split_ids))  # maps split names to ids (for columns in data frames)

        aggrs = []  # list of aggregation tuples, in same order as in the predict-clause
        aggr_ids = []  # ids for columns of fields to aggregate. Same order as in predict-clause

        basenames = set(split_names)  # set of names of fields needed for basemodel of this query
        for t in predict:
            if isinstance(t, str):
                # t is a string, i.e. name of a field that is split by
                name = t
                predict_names.append(name)
                try:
                    predict_ids.append(split_name2id[name])
                except KeyError:
                    raise ValueError("Missing split-tuple for a split-field in predict: " + name)
                basenames.add(name)
            else:
                # t is an aggregation/density tuple
                # TODO: test this
                if t[NAME_IDX] == 'model vs data':
                    raise ValueError("Aggregations or Density queries on 'model vs data'-field are not possible")
                id_ = _tuple2str(t) + next(idgen)
                aggrs.append(t)
                aggr_ids.append(id_)
                predict_names.append(_tuple2str(t))  # generate column name to return
                predict_ids.append(id_)
                basenames.update(t[NAME_IDX])

        basemodel = self.copy().model(basenames, where, self.name + '_base')

        # (2) derive a sub-model for each requested aggregation
        splitnames_unique = set(split_names)

        # for density: keep only those fields as requested in the tuple
        # for 'normal' aggregations: remove all random variables of other measures which are not also
        # a used for splitting, or equivalently: keep all random variables of dimensions, plus the one
        # for the current aggregation

        def _derive_aggregation_model(aggr):
            #aggr_model = basemodel.copy()
            aggr_model = basemodel.copy(name=next(aggr_model_id_gen))
            if aggr[METHOD_IDX] == 'density':
                return aggr_model.model(model=aggr[NAME_IDX])
            else:
                return aggr_model.model(model=list(splitnames_unique | set(aggr[NAME_IDX])))

        aggr_model_id_gen = utils.linear_id_generator(prefix=self.name + "_aggr")
        aggr_models = [_derive_aggregation_model(aggr) for aggr in aggrs]

        # (3) generate input for model aggregations,
        # i.e. a cross join of splits of all dimensions
        if len(splitby) == 0:
            input_frame = pd.DataFrame()
        else:
            def _get_group_frame(split, column_id):
                # could be 'model vs data'
                # TODO: does that really work? should I rather 'pimp' the 'byname' function? would that be consistent?
                field = basemodel._modeldata_field if split[NAME_IDX] == 'model vs data' \
                    else basemodel.byname(split[NAME_IDX])
                # TODO: replace with self.extent
                domain = field['domain'].bounded(field['extent'])
                try:
                    splitfct = sp.splitter[split[METHOD_IDX].lower()]
                except KeyError:
                    raise ValueError("split method '" + split[METHOD_IDX] + "' is not supported")
                frame = pd.DataFrame({column_id: splitfct(domain.value(), split[ARGS_IDX])})
                frame['__crossIdx__'] = 0  # need that index to cross join later
                return frame

            def _crossjoin(df1, df2):
                return pd.merge(df1, df2, on='__crossIdx__', copy=False)

            # TODO: speedup by first grouping by 'model vs data'?
            group_frames = map(_get_group_frame, splitby, split_ids)
            input_frame = reduce(_crossjoin, group_frames, next(group_frames)).drop('__crossIdx__', axis=1)

        # (4) query models and fill result data frame
        """ question is: how to efficiently query the model? how can I vectorize it?
            I believe that depends on the query. A typical query consists of
            dimensions for splits and then aggregations and densities.
            For the case of aggregations a conditioned model has to be
            calculated for every split. I don't see how to vectorize / speed
            this up easily.
            For densities it might be very well possible, as the splits are
            now simply input to some density function.

            it might actually be faster to first condition the model on the
            dimensions (values) and then derive the measure models...
            note: for density however, no conditioning on the input is required
        """

        # build list of comparison operators, depending on split types. Needed to condition on each tuple of the input
        #  frame when aggregating
        method2operator = {
            "equidist": "==",
            "equiinterval": "in",
            "identity": "==",
            "elements": "=="
        }
        operator_list = [method2operator[method] for (_, method, __) in splitby]

        result_list = [pd.DataFrame()]
        for idx, aggr in enumerate(aggrs):
            aggr_results = []
            aggr_model = aggr_models[idx]
            if aggr[METHOD_IDX] == 'density':
                # TODO (1): this is inefficient because it recalculates the same value many times, when we split on more
                #  than what the density is calculated on
                # TODO: to solve it: calculate density only on the required groups and then join into the result table.
                # TODO (2): .density also returns the frequency of the specified observation. Calculating it like this
                #  is also super inefficient, since we really just need to group by all of the split-field and count
                #  occurrences - instead of iteratively counting occurrences, which involves a linear scan on the data
                #  frame for each call to density
                # TODO (3) we even need special handling for the model/data field. See the density method.
                # this is much too complicated. Somehow there must be an easier way to do all of this...
                try:
                    # select relevant columns and iterate over it
                    names = aggr[NAME_IDX]
                    ids = [split_name2id[name] for name in names]
                    # if we split by 'model vs data' field we have to add this to the list of column ids
                    if 'model vs data' in splitnames_unique:
                        ids.append(split_name2id['model vs data'])
                        # names = list(aggr[NAME_IDX])
                        names.append('model vs data')
                except KeyError as err:
                    raise ValueError("missing split-clause for field '" + str(err) + "'.")
                subframe = input_frame[ids]
                for _, row in subframe.iterrows():
                    res = aggr_model.density(names, row)
                    aggr_results.append(res)

                # TODO: the normalization is incorrect: it must not normalize to 1, but to some slice
                # normalize data frequency to data probability
                # get view on data. there is two distinct cases that we need to worry about
                aggr_results = pd.Series(aggr_results)
                if 'model vs data' in splitnames_unique:
                    # case 1: we split by 'model vs data'. need to select only the 'data' rows
                    id = split_name2id['model vs data']
                    mask = input_frame[id] == 'data'
                    if mask.any():
                        sum = aggr_results.loc[mask].sum()
                        aggr_results.loc[mask] /= sum
                elif basemodel.mode == 'data':
                    # case 2: we filter 'model vs data' on 'data'
                    aggr_results = aggr_results / aggr_results.sum()

            else:  # it is some aggregation
                if len(splitby) == 0:
                    # there is no fields to split by, hence only a single value will be aggregated
                    # i.e. marginalize all other fields out
                    # TODO: can there ever be more fields left than the one(s) that is aggregated over?
                    if len(aggr[NAME_IDX]) != aggr_model.dim:  # debugging
                        raise ValueError("uh, interesting - check this out, and read the todo above this code line")
                    singlemodel = aggr_model.copy().marginalize(keep=aggr[NAME_IDX])
                    res = singlemodel.aggregate(aggr[METHOD_IDX], opts=aggr[ARGS_IDX+1])
                    # reduce to requested dimension
                    i = singlemodel.asindex(aggr[YIELDS_IDX])
                    aggr_results.append(res[i])
                else:
                    row_id_gen = utils.linear_id_generator(prefix="_row")
                    for row in input_frame.itertuples(index=False, name=None):
                        pairs = zip(split_names, operator_list, row)
                        # derive model for these specific conditions
                        rowmodel_name = aggr_model.name + next(row_id_gen)
                        rowmodel = aggr_model.copy(name=rowmodel_name).condition(pairs).marginalize(keep=aggr[NAME_IDX])
                        res = rowmodel.aggregate(aggr[METHOD_IDX], opts=aggr[ARGS_IDX+1])
                        # reduce to requested dimension
                        i = rowmodel.asindex(aggr[YIELDS_IDX])
                        aggr_results.append(res[i])

            # generate DataSeries from it
            columns = [aggr_ids[idx]]
            df = pd.DataFrame(aggr_results, columns=columns)
            result_list.append(df)

        # QUICK FIX: when splitting by 'equiinterval' we get intervals instead of scalars as entries
        # however, I cannot currently handle intervals on the client side easily
        # so we just turn it back into scalars
        def mean(entry):
            return (entry[0]+entry[1])/2
        column_interval_list = [split_name2id[name] for (name, method, __) in splitby if method == 'equiinterval']
        for column in column_interval_list:
            input_frame[column] = input_frame[column].apply(mean)

        # (5) filter on aggregations?
        # TODO? actually there should be some easy way to do it, since now it really is SQL filtering

        # (6) collect all results into data frames
        result_list.append(input_frame)
        data_frame = pd.concat(result_list, axis=1)
        # (7) get correctly ordered frame that only contain requested fields
        data_frame = data_frame[predict_ids]  # flattens
        # (8) rename columns to be readable (but not unique anymore)
        data_frame.columns = predict_names

        return (data_frame, basemodel) if returnbasemodel else data_frame

    def select_data(self, what, where=[], opts=None):
        mask = [True]*len(self.data)
        # iteratively build boolean selection mask
        for (name, operator, values) in where:
            operator = operator.lower()
            column = self.data[name]
            if operator == 'in':
                dtype = self.byname(name)['dtype']
                if dtype == 'numerical':
                    # df = df.loc[column.between(*values, inclusive=True)]
                    mask &= column.between(*values, inclusive=True)
                elif dtype == 'string':
                    # df = df.loc[column.isin(values)]
                    mask &= column.isin(values)
                else:
                    raise TypeError("unsupported field type: " + str(dtype))
            else:
                # values is necessarily a single scalar value, not a list
                if operator == 'equals' or operator == '==':
                    # df = df.loc[column == values]
                    mask &= column == values
                elif operator == 'greater' or operator == '>':
                    # df = df.loc[column > values]
                    mask &= column > values
                elif operator == 'less' or operator == '<':
                    # df = df.loc[column < values]
                    mask &= column < values
                else:
                    raise ValueError('invalid operator for condition: ' + str(operator))
        return self.data.loc[mask, what]

    def select(self, what, where=[], opts=None):
        """Returns the selected attributes of all data items that satisfy the conditions as a
        pandas DataFrame.
        By default it selects data only, i.e. it will not return any samples from the model. It may, however, also
        sample from the model if:
          * the where-clause implies so (i.e. contains a filter of the 'model vs data'-field on the value 'model'.
          * the what-clause contains teh 'model vs data'-field (both, data and samples, will be returned)
        """
        # check for empty queries
        if len(what) == 0:
            return pd.DataFrame()

        # if 'model vs data' is present as a filter the filter is respected: i.e. either one of them or both data is
        #  generated
        # iff 'model vs data' is present in what the column will be returned
        # if 'model vs data' is not present in any of the arguments it defaults to a data-select, i.e. a filter for data

        # remove 'model vs data' from what
        pure_what = [w for w in what if w != 'model vs data']

        # check that all columns to select are in data
        if any((label not in self.data.columns for label in pure_what)):
            raise KeyError('at least on of ' + str(pure_what) + ' is not a column label of the data.')

        # remove 'model vs data' from filters
        mvd_filter = None
        pure_where = []
        for w in where:
            if w[NAME_IDX] == 'model vs data':
                mvd_filter = w
            else:
                pure_where.append(w)

        # set default filter for 'model vs data'. Do not if 'model vs data' is in what, because that implies the query
        #  wants both
        if mvd_filter is None and 'model vs data' not in what:
            mvd_filter = ConditionTuple('model vs data', "==", "data")

        # TODO: !!!!!!!! !!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!!! #
        # DEBUG/DEVELOP: always set it to data only
        mvd_filter = ConditionTuple('model vs data', "==", "data")

        # now infer mode
        mode = dm.DiscreteDomain(['model', 'data'])
        if mvd_filter is not None:
            mode.apply(mvd_filter.operator, mvd_filter.value)
        mode = mode.values()

        # select data and or sample, and add model vs data column if requested
        if 'model' in mode:
            # todo: does not meet conditions...!!
            # todo: sampling from less requires model marginalization?
            # todo: default number of samples?
            samples = self.sample(100)
            if 'model vs data' in what:
                samples['model vs data'] = 'model'
        if 'data' in mode:
            data = self.select_data(pure_what, pure_where)
            if 'model vs data' in what:
                data['model vs data'] = 'data'

        # concatenate to one data frame (and prevent copy if possible)
        if 'model' in mode and 'data' in mode:
            df = pd.concat([samples, data], ignore_index=True)
        elif 'model' in mode:
            df = samples
        else:
            df = data

        # return reordered
        return df.loc[:, what]

    def generate_model(self, opts={}):
        # call specific class method
        callbacks = self._generate_model(opts)
        self._update_all_field_derivatives()
        for callback in callbacks:
            callback()
        return self

    def _generate_data(self, opts=None):
        """Provided that self is a functional model, this method it samples opts['n'] many samples from it
        and sets it as the models data. It also sets the 'mode' of this model to 'both'. If not n is given 1000
        samples will be created

        """
        if opts is None:
            opts = {}
        n = opts['n'] if 'n' in opts else 1000

        if self.mode != "model" and self.mode != "both":
            raise ValueError("cannot sample from empty model, i.e. the model has not been learned/set yet.")

        self.data = self.sample(n)
        self.mode = 'both'
        return self


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import data.crabs.crabs as crabs
    import data.iris.iris as iris

    import cond_gaussians as cg

    irisdata = iris.mixed('data/iris/iris.csv')
    crabsdata = crabs.mixed('data/crabs/australian-crabs.csv')

    iris = cg.ConditionallyGaussianModel("iris").fit(irisdata)

    print(iris.name)
