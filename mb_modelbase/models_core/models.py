# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

This module defines:

   * Model: an abstract base class for models.
   * Field: a class that represent random variables in a model.
   * Condition, Split, Aggregation: convenience constructor functions to easily make suitable clauses for PQL queries
"""
import copy as cp
from collections import namedtuple
from functools import reduce
from operator import mul
import pickle as pickle
import numpy as np
import pandas as pd
import logging

from mb_modelbase.models_core import domains as dm
from mb_modelbase.models_core import splitter as sp
from mb_modelbase.utils import utils as utils
from mb_modelbase.models_core import data_aggregation as data_aggr
from mb_modelbase.models_core import data_operations as data_ops

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

"""Internal utility functions: For a number of interfaces we want to allow both: single name of fields or single fields, and sequences of these. These function help to get them in a uniform way internally: sequences of names."""


def _name_from_field(base):
    """Base may either be a field name or a Field. It returns the fields name in either cases."""
    if isinstance(base, str):
        return base
    try:
        return base['name']
    except KeyError:
        raise TypeError('Base must be a Field-dict or a name')


def _is_single_name_or_field(obj):
    return isinstance(obj, (str, dict))


def _to_sequence(obj):
    return [obj] if _is_single_name_or_field(obj) else obj


def _to_name_sequence(obj):
    return list(map(_name_from_field, _to_sequence(obj)))


def Field(name, domain, extent, dtype='numerical'):
    """ A constructor that returns 'Field'-dicts, i.e. a dict with three components
        as passed in:
            'name': the name of the field
            'domain': the domain of the field,
            'extent': the extent of the field that will be used as a fallback for domain if domain is unbounded but a
                value for domain is required
            'dtype': the data type that the field represents. Possible values are: 'numerical' and 'string'
    """
    if not extent.isbounded():
        raise ValueError("extents must not be unbounded")
    if dtype not in ['numerical', 'string']:
        raise ValueError("dtype must be 'string' or 'numerical'")
    field = {'name': name, 'domain': domain, 'extent': extent, 'dtype': dtype, 'hidden': False, 'default_value': None}
    return field


def Aggregation(base, method='maximum', yields=None, args=None):
    name = _to_name_sequence(base)
    if yields is None:
        yields = "" if method == 'density' or method == 'probability' else name[0]
    if args is None:
        args = []
    return AggregationTuple(name, method, yields, args)


def Condition(base, operator, value):
    return ConditionTuple(_name_from_field(base), operator, value)


def Density(base):
    return Aggregation(base, method='density')


def Probability(base):
    return Aggregation(base, method='probability')


def Split(base, method=None, args=None):
    if isinstance(base, str):
        name = base
        if method is None:
            raise ValueError('Cannot infer suitable split method if only a name (but not a Field) is provided.')
    else:
        try:
            name = base['name']
        except KeyError:
            raise TypeError('Base must be a Field-dict or a name')
        if method is None:
            if base['dtype'] == 'string':
                method = 'elements'
            elif base['dtype'] == 'numerical':
                method = 'equiinterval'
            else:
                raise ValueError('unknown dtype: ' + str(base['dtype']))
    if args is None:
        if method == 'equiinterval' or method == 'equidist':
            args = [25]
        elif method == 'identity' or method == 'elements':
            args = []
        else:
            raise ValueError('unknown method type: ' + str(method))
    else:
        # TODO: this whole way of handling args sucks... I should replace it with a default of None and a dict for actual values...
        # promote non-iterable to sequences
        if not isinstance(args, str):
            try:
                iter(args)  # if that did fail it's not iterable and we don't need to promote it
            except TypeError:
                args = [args]
    return SplitTuple(name, method, args)


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
               + ("*" if field['domain'].isbounded() else "")  # * marks fields with bounded domains

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


def split_training_test_data(df):
    # select training and test data
    limit = int(min(max(df.shape[0]*0.05, 25), 50, df.shape[0]))  # 5% of the data, but not less than 25 and not more than 50
    test_data = df.iloc[:limit, :]
    data = df.iloc[limit:, :]
    return test_data, data


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
        mi, ma = column.min(), column.max()
        d = (ma-mi)*0.1
        field = Field(colname, dm.NumericDomain(), dm.NumericDomain(mi-d, ma+d), 'numerical')
        fields.append(field)
    return fields

def to_category_cols(df, colnames):
    """Returns df where all columns with names in colnames have been converted to the category type using pd.astype('category'.
    """
    # df.loc[:,colnames].apply(lambda c: c.astype('category'))  # also works, but more tedious merge with not converted df part
    for c in colnames:
        # .cat.codes access the integer codes that encode the actual categorical values. Here, however, we want such integer values.
        df[c] = df[c].astype('category').cat.codes
    return df


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
        It silently drops any duplicate names.
        """
        assert (len(set(names)) <= len(self.names))
        return utils.sort_filter_list(names, self.names)

    def __init__(self, name="model"):
        self.name = name
        # the following is all done in _fields_set_empty, which is called below
        # self.fields = []
        # self.names = []
        # self.extents = []
        # self.dim = 0
        # self._name2idx = {}
        self._fields_set_empty()
        self.data = pd.DataFrame([])
        self.test_data = pd.DataFrame([])
        self._aggrMethods = None
        self.mode = None
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

    def set_model_params(self, **kwargs):
        """Sets explicitly the parameters of a model.

        This method has the following effects:
         - the models mode is set to 'model'
         - the models fields attribute is derived accordingly
        """
        if self.mode != 'empty':
            raise ValueError("cannot set parameters on non-empty model.")

        callbacks = self._set_model_params(**kwargs)
        self._update_all_field_derivatives()
        self.mode = "model"
        for callback in callbacks:
            callback()

        return self

    def _set_model_params(self, **kwargs):
        """ Sets the parameters of a model. See also model.set_model_params.

        This method must be implementedby all actual model classes. Make sure it complies to the requirements, see
        model.set_model_params.
        """
        raise NotImplementedError("You have to implement the _set_model_params method in your model!")

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

    def _set_data_mixed(self, df, silently_drop, num_names=None, cat_names=None):
        """see Model._set_data"""

        # split in categorical and numeric columns
        if num_names is None or cat_names is None:
            _, cat_names, num_names = get_columns_by_dtype(df)
        self._categoricals = cat_names
        self._numericals = num_names

        # check if dtype are ok
        #  ... nothing to do here ...

        # set data for model as needed
        #  reorder data frame such that categorical columns are first
        df = df[self._categoricals + self._numericals]

        # shuffle and set data frame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_data, self.data = split_training_test_data(df)

        # derive and set fields
        self.fields = get_discrete_fields(df, self._categoricals) + \
                      get_numerical_fields(df, self._numericals)

        # normalize numerical part of data frame. For better numerical performance.
        # this is done at model level
        # if normalize:
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

        # shuffle and set data frame
        df = pd.DataFrame(df, columns=numericals).sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_data, self.data = split_training_test_data(df)

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

        # shuffle and set data frame
        df = pd.DataFrame(df, columns=categoricals).sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_data, self.data = split_training_test_data(df)

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

        if self.data is None and self.mode != 'data':
            raise ValueError(
                'No data frame to fit to present: pass it as an argument or set it before using set_data(df)')

        try:
            callbacks = self._fit(**kwargs)
            self.mode = "both"
            #self._update_all_field_derivatives()
            for callback in callbacks:
                callback()
        except NameError:
            raise NotImplementedError("You have to implement the _fit method in your model!")
        return self

    def marginalize(self, keep=None, remove=None, is_pure=False):
        """Marginalizes random variables out of the model. Either specify which
        random variables to keep or specify which to remove.

        Note that marginalization depends on the domain of a random variable. That is: if
        its domain is bounded it is conditioned on this value (and marginalized out).
        Otherwise it is 'normally' marginalized out (assuming that the full domain is available).

        Arguments:
            keep: A sequence or a sequence of names of dimensions or fields of a model. The given dimensions of this model are kept. All other random variables are marginalized out.
            remove: A sequence or a sequence of names of dimensions or fields of a model. The given dimensions of this model are marginalized out.
            is_pure: An performance optimization parameter. Set to True if you can guarantee that keep and remove do not contain the special value "model vs data".

        Returns:
            The modified model.
        """
        # logger.debug('marginalizing: ' + ('keep = ' + str(keep) if remove is None else ', remove = ' + str(remove)))

        if keep is not None and remove is not None:
            raise ValueError("You may only specify either 'keep' or 'remove', but not both.")

        if keep is not None:
            keep = _to_name_sequence(keep)
            if keep == '*':
                keep = self.names
            elif not is_pure:
                keep = [name for name in keep if name != 'model vs data']
            if not self.isfieldname(keep):
                raise ValueError("invalid random variables names in argument 'keep': " + str(keep))
        elif remove is not None:
            remove = _to_name_sequence(remove)
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

        This is the internal version of the public method marginalize. You probably want to call that one, not _marginalize.

        Args:
            keep: the names of the fields to keep, in the same order than in self.fields.
            remove: the names of the fields to remove, in the same order than in self.fields.

        Notes:
            You have to specify at least one of keep and remove.
            Neither keep nor remove may contain the field 'model vs data'.
        """
        if keep is not None and remove is not None:
            assert (set(keep) | set(remove) == set(self.names))
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
            # Note: we never need to copy data, since we never change data. creating views is enough
            self.data = self.data.loc[:, keep]
            self.test_data = self.test_data.loc[:, keep]
            if self.mode == 'data':
                self._update_remove_fields(remove)
                return self

        # Model marginalization
        # we only get here, if self.mode != 'data
        # there are three cases of marginalization:
        # (1) unrestricted domain, (2) restricted, but not singular domain, (3) singular domain
        # we handle case (2) and (3) in ._conditionout, then case (1) in ._marginalizeout
        if len(cond_out) != 0:
            callbacks = self._conditionout(self.inverse_names(cond_out, sorted_=True), cond_out)
            self._update_remove_fields(cond_out)
            for callback in callbacks:
                callback()
            remove = self.inverse_names(keep, sorted_=True)  # do it again, because fields may have changed

        if len(keep) != self.dim and not self._isempty():
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

    def condition(self, conditions=None, is_pure=False):
        """Conditions this model according to the list of three-tuples
        (<name-of-random-variable>, <operator>, <value(s)>). In particular
        objects of type ConditionTuples are accepted and see there for allowed values.

        Note: This only restricts the domains of the random variables. To
        remove the conditioned random variable you need to call marginalize
        with the appropriate parameters.

        Returns:
            The modified model.
        """
        if conditions is None:
            conditions = []

        if isinstance(conditions, tuple):
            conditions = [conditions]

        # TODO: simplify the interface?

        # can't do that because I want to allow a zip object as conditions...
        # if len(conditions) == 0:
        #     return self

        # treat 'model vs data' field correctly. It must be applied first.
        names = []
        if is_pure:
            pure_conditions = conditions
        else:
            # we allow conditioning on 'model vs data' field:
            # find 'model vs data' field and apply conditions
            pure_conditions = []
            for condition in conditions:
                (name, operator, values) = condition
                if name == 'model vs data':
                    # TODO: need clean handling of such meta-attribute/fields
                    # TODO: right now there seems no clean sync between self.mode and self._modeldata_field
                    # TODO: I think self.mode should not be assignable from outside?
                    # allowed = self._modeldata_field['domain'].values()
                    # if values not in allowed:
                    #     raise ValueError("conditioning ", + str(allowed) + " on " + str(values) + " impossile!")
                    # if self.mode == 'both':
                    #     self.mode = values
                    # elif self.mode == values:
                    #     pass
                    # else:
                    #     raise ValueError("conditioning ", + str(self.mode) + " on " + str(values) + " impossile!")
                    domain = self._modeldata_field['domain']
                    domain.apply(operator, values)
                    # set internal mode accordingly to both, data or model
                    value = domain.value()
                    self.mode = value if value == 'model' or value == 'data' else 'both'
                else:
                    names.append(name)
                    pure_conditions.append(condition)

        # condition the domain of the fields
        for (name, operator, values) in pure_conditions:
            self.byname(name)['domain'].apply(operator, values)

        # condition model
        # todo: currently, conditioning of a model is always only done when it is conditioned out.
        # in the future this should be decided by the model itself:
        #  for an emperical model it is always possible to compute the conditional model
        #  for an gaussian model we may compute it for point conditinals right away, but we maybe would not want to do it for range conditionals

        # condition data
        if self.mode != 'model':  # i.e. == 'data' or == 'both'
            self.data = data_ops.condition_data(self.data, pure_conditions)
            self.test_data = data_ops.condition_data(self.test_data, pure_conditions)
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


    def hide(self, dims, val=True):
        """Hides the given fields. Provide either a single field or a field name or a sequence of these.

        Only fields that have a default value set can be hidden.

        Hiding a dimension does not cause any actual change in the model. However, it allows you to query to model without specifying a value for the hidden dimension. In this sense, it provides you with a view on a splice of the model, where hidden dimensions are fixed to their default values. That is why only dimensions with default values can be hidden.

        Note that any hidden dimension is still reported as a dimension of the model by means of auxilary methods like Model.byname, Model.isfieldname, Model.fields, etc etc.

        However, hiding a dimension with name 'foo' has the following effects on subsequent queries against the model:
          * density: No value for 'foo' needs to be specified, but the default value is used and the density at that point is returnd
          * probability: TODO!??
          * predict: Predictions are done 'normally', i.e. including the hidden dimension foo. However, in the returned tuples the values for foo are removed
          * sample: Sampling is done 'normally', but the hidden dimensions are removed from the generated samples.
          * condition: no issue here what so ever. It is possible to condition hidden dimensions.
          * marginalize: no issue here either.

        For more details, look at the documentation of each of the methods.
        """
        dims = _to_name_sequence(dims)
        for field in self.byname(dims):
            if field['default_value'] is None:
                raise ValueError("Cannot hide field '" + field['name'] + "' that has no default value set.")
            self._hidden_count -= field['hidden'] - val
            field['hidden'] = val

        return self

    def reveal(self, dims):
        """Reveals the given fields. Provide either a single field or a field name or a sequence of these.

        See also Model.hide().
        """
        return self.hide(dims, False)

    def set_default(self, dims, values=None):
        """Sets default values for dimensions. There is two ways of specifying the arguments:

        1. Provide a dict of <field-name> to <default-value> (as first argument or argument dims).
        2. Provide two sequences: dims and values that contain the field names and default values respectively.

        You may set default values for hidden and non-hidden dimensions of the model.

        Returns self.
        """

        if values is None:
            # dims is a dict of <field-name> to <default-value>
            if not isinstance(dims, dict):
                raise TypeError("dims must be a dict of <field-name> to <default-value>, if values is None.")
        else:
            # dims and values are two lists
            if len(dims) != len(values):
                raise ValueError("dims and values must be of equal length.")
            dims = dict(zip(dims, values))

        if not self.isfieldname(dims.keys()):
            raise ValueError("dimensions must be specified by their name and must be a dimension of this model")

        # update default values
        for name, value in dims.items():
            field = self.byname(name)
            if not field['domain'].contains(value):
                raise ValueError("The value to set as default must be within the domain of the dimension.")
            field['default'] = value

        return self

    def freeze(self, dims, values=None):
        """Freeze given dimensions to given values."""
        self.set_default(dims, values)
        if values is None:
            dims = dims.keys()
        self.hide(dims)

    def aggregate_data(self, method, opts=None):
        """Aggregate the models data according to given method and options, and returns this aggregation."""
        return data_aggr.aggregate_data(self.data, method, opts)
        #data_aggr.aggregate_data(self.test_data, method, opts)  # TODO: should I add the option to compute aggregations on a selectable part of the data

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
            The aggregation of the model as a sequence.
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

    def density(self, values=None, names=None):
        """Returns the density at given point.

         Args:
            There are several ways to specify the arguments:
            (1) A list of values in <values> and a list of dimensions names in <names>, with corresponding order. They need to be in the same order than the dimensions of this model.
            (2) A list of values in <values> in the same order than the dimensions of this model. <names> must not be passed. This is faster than (1).
            (3) A dict of key=name:value=value in <values>. <names> is ignored.

        Special field 'model vs data':

            * you must not use option (1) # TODO?
            * if you use option (2): supply it as the last element of the domain list
            * if you use option (3): use the key 'model vs data' for its domain

        You may pass a special field with name 'model vs data':
          * If its value is 'data' then it is interpreted as a density query against the data, i.e. frequency of items is returned.
          * If the value is 'model' it is interpreted as a query against the model, i.e. density of input is returned.
          * If value is 'both' a 2-tuple of both is returned.

        Hidden dimensions:

        The density will be computed on the model over all, hidden and non-hidden dimensions. If you do not specify a value for a hidden dimension its default value will be used instead to compute the density at that point.

        If you use variant (1) and (3) of specifying arguments:
        You may or not provide a value for hidden dimensions. You may also 'mix', i.e. give values for some hidden dimensions but not for other hidden dimensions.

        If you use variant (2):
        You must not give any value of hidden dimensions. It would be impossible to determine which you specified and which not.
        """
        if self._isempty():
            raise ValueError('Cannot query density of 0-dimensional model')

        # normalize parameters to ordered list form
        mode = self.mode
        if isinstance(values, dict):
            # dict was passed
            if len(values) - ('model vs data' in values) != self.dim:
                raise ValueError('Incorrect number of values given')
            values = [values[name] for name in self.names]
            mode = values.get('model vs data', self.mode)
        elif names is not None and values is not None:
            # unordered list was passed in. Not model vs data may be passed
            if len(values) != len(names):
                raise ValueError('Length of names and values does not match.')
            elif len(set(names)) == len(names):
                raise ValueError('Some name occurs twice.')
            sorted_ = sorted(zip(self.asindex(names), values), key=lambda pair: pair[0])
            values = [pair[1] for pair in sorted_]
        elif len(values) == self.dim:
            # correctly ordered list was passed in, without model vs data domain
            pass
        elif len(values) == self.dim + 1:
            # correctly ordered list was passed in, but with model vs data domain
            mode = values[-1]
            values = values[:-1]
        else:
            raise ValueError("Invalid number of values passed.")

        # data frequency
        if mode == "both" or mode == "data":
            # TODO?? should I do this against the test data as well?? expand mode to test data or something? also applies to similar sections in model.probability
            #cnt = len(data_ops.condition_data(self.data, zip(self.names, ['=='] * self.dim, values)))
            cnt = data_ops.density(self.data, values)
            if mode == "data":
                return cnt

        #p = self._density(data_ops.reduce_to_scalars(values))
        p = self._density(values)
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

    def probability(self, domains=None, names=None):
        """
        Returns the probability of given event.

        By default this returns an approximation to the true probability. To implement it exactly or a different approximation for your model class reimplement the method _probability.

        Args:
            There are several ways to specify the arguments:
            (1) A list of domains in <domains> and a list of dimensions names in <names>, with corresponding order. They need to be in the same order than the dimensions of this model.
            (2) A list of domains in <domains> in the same order than the dimensions of this model. <names> must not be passed. This is faster than (1).
            (3) A dict of key=name:value=domain in <domains>. <names> is ignored.

        You may pass a special field with name 'model vs data':
          * you cannot use option (1) # TODO?
          * if you use option (2): supply it as the last element of the domain list
          * if you use option (3): use the key 'model vs data' for its domain
          * If its value is 'data' then it is interpreted as a density query against the data, i.e. frequency of items is returned.
          * If the value is 'model' it is interpreted as a query against the model, i.e. density of input is returned.
          * If value is 'both' a 2-tuple of both is returned.
        """
        if self._isempty():
            raise ValueError('Cannot query density of 0-dimensional model')

        # normalize parameters to ordered list form
        mode = self.mode
        if isinstance(domains, dict):
            # dict was passed
            if len(domains) - ('model vs data' in domains) != self.dim:
                raise ValueError('Incorrect number of values given')
            domains = [domains[name] for name in self.names]
            mode = domains.get('model vs data', [self.mode])[0]
        elif names is not None and domains is not None:
            if len(domains) != len(names):
                raise ValueError('Length of names and values does not match.')
            elif len(set(names)) == len(names):
                raise ValueError('Some name occurs twice.')
            # unordered list was passed in. Not model vs data may be passed
            sorted_ = sorted(zip(self.asindex(names), domains), key=lambda pair: pair[0])
            domains = [pair[1] for pair in sorted_]
        elif len(domains) == self.dim:
            # correctly ordered list was passed in, without model vs data domain
            pass
        elif len(domains) == self.dim + 1:
            # correctly ordered list was passed in, but with model vs data domain
            mode = domains[-1][0]
            domains = domains[:-1]
        else:
            raise ValueError("Invalid number of values passed.")

        # data probability
        if mode == "both" or mode == "data":
            data_prob = data_ops.probability(self.data, domains)
            if mode == "data":
                return data_prob

        # model probability
        model_prob = self._probability(domains)
        if mode == "model":
            return model_prob
        elif mode == "both":
            return model_prob, data_prob
        else:
            raise ValueError("invalid value for mode : ", str(mode))

    def _probability(self, domains):
        try:
            cat_len = len(self._categoricals)
        except AttributeError:
            # self._categoricals might not be implemented in a particular model. this is the fallback:
            cat_len = sum(f['dtype'] == 'string' for f in self.fields)
            cat_len = sum(f['dtype'] == 'string' for f in self.fields)
        return self._probability_generic_mixed(domains[:cat_len], domains[cat_len:])

    def _probability_generic_mixed(self, cat_domains, num_domains):
        """
        Returns an approximation to the probability of the given event.
        This works generically for any model mixed model that stores categorical dimensions before numerical
        dimensions. It is assumed that all domains are given in their respective order in the model.

        Args:
            list of domains of the event in correct order. Valid domains are:
              * for categorical columns: a sequence of strings, e.g. ['A'], or ['A','B']
              * for quantitative columns: a 2-element sequence or tuple, e.g. [1,2], or [1,1] or [2,6]
                * if [l,h] is the interval, then neither l nor h may be +-infinity and it must hold l <= h
        """
        # volume of all combined each quantitative domains
        vol = reduce(mul, [high - low for low, high in num_domains], 1)
        # map quantitative domains to their mid
        y = [(high + low) / 2 for low, high in num_domains]

        # sum up density over all elements of the cartesian product of the categorical part of the event
        # TODO: generalize
        assert(all(len(d) == 1 for d in cat_domains))
        x = list([d[0] for d in cat_domains])
        return vol*self._density(x+y)

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
          * data (a reference to it!!!) # CURRENTLY: copied!
          * fields (deep copy)
        """
        name = self.name if name is None else name
        mycopy = self.__class__(name)
        mycopy.data = self.data.copy()
        mycopy.test_data = self.test_data.copy()
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy.mode = self.mode
        mycopy._modeldata_field = cp.deepcopy(self._modeldata_field)
        mycopy._update_all_field_derivatives()
        return mycopy

    def _condition_values(self, names=None, pairflag=False, to_scalar=True):
        """Returns the list of values to condition on given a sequence of random variable names to condition on.
        Essentially, this is a look up in the domain restrictions of the fields to be conditioned on (i.e. the sequence names).

        Args:
            names: sequence of random variable names to get the conditioning domain for. If not given the condition values for all dimensions are returned.
            pairflag = False: Optional. If set True not a list of values but a zip-object of the names and the
             values to condition on is returned
            to_scalar: flag to turn any non-scalar values to scalars.
        """
        if not to_scalar:
            raise NotImplemented
        fields = self.fields if names is None else self.byname(names)
        # It's not obvious but this is aequivalent to the more detailed code below...
        cond_values = [field['domain'].mid() for field in fields]
        # cond_values = []
        # for field in fields:
        #     domain = field['domain']
        #     dvalue = domain.value()
        #     if not domain.isbounded():
        #         raise ValueError("cannot condition random variables with not bounded domain!")
        #     if field['dtype'] == 'numerical':
        #         # TODO: I think this is wrong. I should solve this issue not here, but where the conditioning is actually applied!
        #         cond_values.append(dvalue if domain.issingular() else (dvalue[1] + dvalue[0]) / 2)
        #     elif field['dtype'] == 'string':
        #         # TODO: I actually know how to apply such multi element conditions
        #         cond_values.append(dvalue if domain.issingular() else dvalue[0])
        #     else:
        #         raise ValueError('invalid dtype of field: ' + str(field['dtype']))

        return zip(names, cond_values) if pairflag else cond_values

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
        self._hidden_count = 0
        return

    def _update_extents(self, to_update=None):
        """Updates self.extents of the fields with names in the sequence to_update.
        Updates all if to_update is None.
        """
        if to_update is None:
            self.extents = [field['domain'].bounded(field['extent']) for field in self.fields]
        else:
            to_update_idx = self.asindex(to_update)
            for idx in to_update_idx:
                field = self.fields[idx]
                self.extents[idx] = field['domain'].bounded(field['extent'])
            return self

    def _update_name2idx_dict(self):
        """Updates (i.e. recreates) the _name2idx dictionary from current self.fields."""
        self._name2idx = dict(zip([f['name'] for f in self.fields], range(self.dim)))

    def _update_remove_fields(self, to_remove=None):
        """Removes the fields in the sequence to_remove (and correspondingly derived structures) from self.fields.
         Removes all if to_remove is None
         @fields
        """
        if to_remove is None:
            return self._fields_set_empty()

        to_remove_idx = self.asindex(to_remove)
        # assert order of indexes
        assert (all(to_remove_idx[i] <= to_remove_idx[i + 1] for i in range(len(to_remove_idx) - 1)))

        for idx in reversed(to_remove_idx):
            self._hidden_count -= self.fields[idx]['hidden']
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
        self._update_extents()
        self._hidden_count = sum(map(lambda f: f['hidden'], self.fields))
        return self

    def model(self, model='*', where=None, as_=None):
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

    def predict(self, predict, where=None, splitby=None, returnbasemodel=False):
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

        if isinstance(predict, (str, tuple)):
            predict = [predict]
        
        if isinstance(where, tuple):
            where = [where]
        
        if isinstance(splitby, tuple):
            splitby = [splitby]
        
        if self._isempty():
            return pd.DataFrame()

        if where is None:
            where = []
        if splitby is None:
            splitby = []
        # return (pd.DataFrame(), pd.DataFrame()) if mode == 'both' else pd.DataFrame()
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
                and 'model vs data' not in filter_names \
                and self.mode == 'both':
            where.append(Condition('model vs data', '==', 'model'))
            filter_names.append('model vs data')
            splitby.append(Split('model vs data', 'identity', []))
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
                field = basemodel.byname(split[NAME_IDX])
                domain = field['domain'].bounded(field['extent'])
                try:
                    splitfct = sp.splitter[split[METHOD_IDX].lower()]
                except KeyError:
                    raise ValueError("split method '" + split[METHOD_IDX] + "' is not supported")
                frame = pd.DataFrame({column_id: splitfct(domain.values(), split[ARGS_IDX])})
                frame['__crossIdx__'] = 0  # need that index to cross join later
                return frame

            def _crossjoin(df1, df2):
                return pd.merge(df1, df2, on='__crossIdx__', copy=False)

            # filter to tuples of (identity_split, split_id)
            id_tpl = tuple(zip(*((s, i) for s, i in zip(splitby, split_ids) if s[METHOD_IDX] == 'identity')))
            identity_splits, identity_ids = ([], []) if len(id_tpl) == 0 else id_tpl

            # filter to tuples of (data_split, split_id)
            split_tpl = tuple(zip(*((s, i) for s, i in zip(splitby, split_ids) if s[METHOD_IDX] == 'data')))
            data_splits, data_ids = ([], []) if len(split_tpl) == 0 else split_tpl

            # all splits are non-data splits
            if len(data_splits) == 0:
                # TODO: speedup by first grouping by 'model vs data'?
                group_frames = map(_get_group_frame, splitby, split_ids)
                input_frame = reduce(_crossjoin, group_frames, next(group_frames)).drop('__crossIdx__', axis=1)

            # all splits are data and/or identity splits
            elif len(data_splits) + len(identity_splits) == len(splitby):

                # compute input frame according to data splits
                data_split_names = [s[NAME_IDX] for s in data_splits]
                assert(self.mode == 'both')
                #limit = 15*len(data_split_names)  # TODO: maybe we need a nicer heuristic? :)
                # #.drop_duplicates()\ # TODO: would make sense to do it, but then I run into problems with matching test data to aggregations on them in frontend, because I drop them for the aggregations, but not for test data select
                input_frame = self.test_data.loc[:, data_split_names]\
                    .sort_values(by=data_split_names, ascending=True)
                pass
                input_frame.columns = data_ids  # rename to data split ids!

                # add identity splits
                for id_, s in zip(identity_ids, identity_splits):
                    field = basemodel.byname(s[NAME_IDX])
                    domain = field['domain'].bounded(field['extent'])
                    assert(domain.issingular())
                    input_frame[id_] = domain.value()

                # TODO: I do not understand why this reset is necesary, but it breaks if I don't do it.
                input_frame = input_frame.reset_index(drop=True)
            else:
                raise NotImplementedError('Currently mixing data splits with any other splits is not supported.')

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
            "identity": "in",
            "elements": "in",
            "data": "in",
        }
        operator_list = [method2operator[method] for (_, method, __) in splitby]

        result_list = [pd.DataFrame()]
        for idx, aggr in enumerate(aggrs):
            aggr_results = []
            aggr_model = aggr_models[idx]

            # it depends on the split whether it is suitable to compute the density or the probability over the result of it
            # i.e.: if we split into intervals, we need probability (which we will do most of the time)
            # but if we split to points we need density (which would most of the time result in zero density on data)
            # we need separate internal implementations of density and probability, but I think it would be nice
            # if to the outside density accepts also intervals/sets and then just calls probability (and returns a probability)
            # though, it's less clean...

            if aggr[METHOD_IDX] == 'density':
                # TODO (1): this is inefficient because it recalculates the same value many times, when we split on more than what the density is calculated on
                # TODO: to solve it: calculate density only on the required groups and then join into the result table.
                # TODO (2): .density also returns the frequency of the specified observation. Calculating it like this
                #  is also super inefficient, since we really just need to group by all of the split-field and count
                #  occurrences - instead of iteratively counting occurrences, which involves a linear scan on the data
                #  frame for each call to density
                # TODO (3) we even need special handling for the model/data field. See the density method.
                # this is much too complicated. Somehow there must be an easier way to do all of this...
                # by now I think we need a new 'type' of fields: artificial fields. We will anyway need this for
                # hyperparameters like fitting params, etc. And we want a clean and eays and generic way to deal with them
                try:
                    # select relevant columns in correct order and iterate over it
                    names = self.sorted_names(aggr[NAME_IDX])
                    ids = [split_name2id[name] for name in names]
                    # if we split by 'model vs data' field we have to add this to the list of column ids
                    if 'model vs data' in splitnames_unique:
                        ids.append(split_name2id['model vs data'])
                        names.append('model vs data')
                except KeyError as err:
                    raise ValueError("missing split-clause for field '" + str(err) + "'.")
                subframe = input_frame.loc[:,ids]

                for row in subframe.itertuples(index=False):
                    # res = aggr_model.density(values=row)
                    res = aggr_model.probability(domains=row)
                    aggr_results.append(res)

            else:  # it is some aggregation
                if len(splitby) == 0:
                    # there is no fields to split by, hence only a single value will be aggregated
                    # i.e. marginalize all other fields out
                    # TODO: can there ever be more fields left than the one(s) that is aggregated over?
                    if len(aggr[NAME_IDX]) != aggr_model.dim:  # debugging
                        raise ValueError("uh, interesting - check this out, and read the todo above this code line")
                    singlemodel = aggr_model.copy().marginalize(keep=aggr[NAME_IDX])
                    res = singlemodel.aggregate(aggr[METHOD_IDX], opts=aggr[ARGS_IDX + 1])
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
                        res = rowmodel.aggregate(aggr[METHOD_IDX], opts=aggr[ARGS_IDX + 1])
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
            return (entry[0] + entry[1]) / 2

        column_interval_list = [split_name2id[name] for (name, method, __) in splitby if method == 'equiinterval']
        for column in column_interval_list:
            input_frame[column] = input_frame[column].apply(mean)

        # QUICK FIX2: when splitting by 'elements' or 'identity' we get intervals instead of scalars as entries
        column_interval_list = [split_name2id[name] for (name, method, __) in splitby if
                                method == 'elements' or method == 'identity']
        for column in column_interval_list:
            input_frame[column] = input_frame[column].apply(lambda entry: entry[0])

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

    def select_data(self, what, where=None, **kwargs):
        # todo: use update_opts also at other places where appropiate (search for validate_opts)
        opts = utils.update_opts({'data_category': 'training data'}, kwargs, {'data_category': ['training data', 'test data']})
        df = self.data if opts['data_category'] == 'training data' else self.test_data
        return data_ops.condition_data(df, where).loc[:, what]

    def select(self, what, where=None, **kwargs):
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
        if where is None:
            where = []

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
            mvd_filter = Condition('model vs data', "==", "data")

        # TODO: !!!!!!!! !!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!!! #
        # DEBUG/DEVELOP: always set it to data only
        mvd_filter = Condition('model vs data', "==", "data")

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
            data = self.select_data(pure_what, pure_where, **kwargs)
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
    import pandas as pd
    import mb_modelbase as mb

    df = pd.read_csv('../../../mb_data/mb_data/iris/iris.csv')
    m = mb.MixableCondGaussianModel()
    m.fit(df)
    res = m.predict(predict=['sepal_width', mb.Aggregation('sepal_length')], splitby=mb.SplitTuple('sepal_width', 'data', [5]))
    print(res)

    res = m.predict(predict=['sepal_width', 'petal_width', mb.Aggregation(['sepal_length'])], splitby=[mb.SplitTuple('sepal_width', 'data', [5]), mb.SplitTuple('petal_width', 'data', [5])])
    print("\n")
    print(res)


    #res = m.predict(predict=['sepal_width', mb.Aggregation('sepal_length')], splitby=mb.SplitTuple('sepal_width', 'equidist', [5]))
    # print(res)

