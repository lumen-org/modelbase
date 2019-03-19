# Copyright (c) 2017-2018 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

This module defines the abstract base class of all models: `Model`.
This is the base class of all real model classes.

It also provides definitions of other essential data types, such as `Field`, `Aggregation`, `Density`, `Probability`,
`Split`, `Condition`
"""

import copy as cp
import functools
import operator
import pickle as pickle
import numpy as np
import pandas as pd
import logging

from mb_modelbase.models_core import base
from mb_modelbase.models_core.base import Condition, Split, Density
from mb_modelbase.models_core.base import NAME_IDX, METHOD_IDX, YIELDS_IDX, ARGS_IDX, OP_IDX, VALUE_IDX
from mb_modelbase.models_core import splitter as sp
from mb_modelbase.models_core import models_predict
from mb_modelbase.utils import utils
from mb_modelbase.utils import data_import_utils
from mb_modelbase.models_core import data_aggregation
from mb_modelbase.models_core import data_operations
from mb_modelbase.models_core import pci_graph
from mb_modelbase.models_core import auto_extent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    """Given a single name or a list of names of fields, returns
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


class Model:
    """An abstract base model that provides an interface to derive submodels from it or query density and other
    aggregations of it.

    Attributes:

        .data : pd.DataFrame

            The data that is assigned to the model. When `.fit()` is called this data is used to learn the model
            parameters for this model instance. `.data` should only be set using `Model.set_data()`. See also the Data
            section below.

        .dim : int

            The current dimension of the model, i.e. the number of fields in `.fields`.

        .extents : sequence of sequences.

            The sequence of extents of the current fields of the model. Is in the same order like in `.fields`. An
            extent is a 'typical' set of values for a field. It is used, for example, if a bounded set of values is
            required as input from a field. E.g. if we split a quantitative field into k intervals,
            we need a bounded initial interval to split. Here, the extent would be used.

        .fields : sequence of `Field`s

            The See the Fields and Order of Fields sections below.

        .history : dict

            See the History section below.

        .mode : [None, 'model', 'data', 'both']

            The mode holds the current state of the model, and it represents it 'position' in the life-cycle of a model:

                * mode is None: the model was instantiated using the class constructor. The model instance now exists,
                but is not fitted to any data or otherwise even filled its internal parameter to represent a valid
                model at all.

                * mode == 'data': Data was assigned to the model using `.set_data()`, but the no model to represent
                that data has yet been learned. Hence, it is not possible to query the model with `.predict()` or the
                like. Anyway, queries against that data alone are possible.

                * mode == 'model: In contrast to ``mode == 'data'`` no data was assigned to the model,
                but it nevertheless represents a distribution of its class and queries such as `.predict()` can be
                made. This is typcially the case if the model class provides a way of (randomly) generating instances
                of it, see `.generate_model`.

                * mode == 'both': Data was assigned to the model AND the models parameters were fit to that data.
                This is most common state of a model and holds for example after you call `.set_data()` and
                thereafter `.fit()`.

        .name : string

            A unique (not enforced) name of the model.

        .names : sequence of strings

            The sequence of names of the current fields of the model.  Is in the same order like `.fields`.

        .parallel_processing : bool

            A flag that indicates whether certain queries should be executed in parallel on multiple available cores
            or not.

        .pci_graph : None or something


        .test_data : pd.DataFrame

            When data is set to a model using `.set_data()` or indirectly by using `.fit()` it will split the data
            into a training data set and test data set. This is the test data set.

    Methods:

        See the documentation at the method level.

    Concepts:

        Fields:

            A model has a number of fields (aka dimensions or random variables or attributes). The model models a
            probability density function on these fields and allows various queries against this density function.
            The fields of a model are accessible by the attribute `fields`. Alternatively, you may access fields by
            name with `Model.byname()`. There is other conversion routines, like `Model.asindex()`,
            `Model.inverse_names()`, `Model.isfieldname()`. Note that when a model shrinks due to marginalization the
            marginalized fields are removed. Hence, `.fields` always represents the 'current' Fields.

        Order of Fields:

            The fields of a model have a particular order that does not change during its lifetime. This is the order
            of fields in the attribute `Model.fields`. Whenever the model returns a list of values or objects related
            to fields, they are in the same order like the fields of the model (unless noted otherwise).

        Data:

            The model is based on data (evidence), and the data can be queried in the same way like the model.

        Hiding/Revealing fields:

            Fields of the model may be hidden, i.e. the values of hidden fields are removed from any output generated
            by a query against the model. See `Model.hide()`

        Default values and Default subset:

            The fields of a model may have a default value and a default subset. These can be set and removed using
            `.set_default_value()` and `.set_default_subset()`, respectively.

            Default values and default subsets (i.e. ranged) are used whenever a query against the model requires
            scalar or range input, respectively, but non is provided in the queries arguments. In these cases the
            set default values or subsets are used.

            The combination of hiding and setting a default value it provides you with a view on a slice of the
            model, where hidden fields are fixed to their default values / subsets.

        History:

            Models also provide a 'history', i.e. information about what operations (condition, marginalize) have
            been applied in order to arrive to the current model. This information is stored under the .history
            attribute and is a organized as a dictionary. The key is the name of the field that was changed. The
            value is a dict with these keys:
                * 'conditioned': a list of conditions applied in that order
                * 'marginalized': a string:
                    either None (if not marginalized in any way), or 'marginalized_out' (if marginalized over the full
                    domain) or 'conditioned_out' (if marginalized after conditioning).

        Model Queries:

            Once the model is initialized (i.e. its `.mode` equals 'model' or 'both') you may run model queries
            against it. There is two types of queries:
                1. modeling queries, i.e. queries that result in an modified model. These are:
                    * marginalization: `.marginalize()`,
                    * conditioning: `.condition()`, and
                    * a complex query method that combines both above (and more): `.model()`
                2. 'execution' queries, i.e. queries that return a specific characterization of the model. These are:
                    * query density: `.density()` which returns the density at a certain point in the domain of the
                        model,
                    * query probability: `.probability()` which returns the probability of a certain event,
                    * sampling: `.sample()` which draws samples according to the distribution,
                    * aggregations: `.aggregate()` which aggregates the model using to chosen method, and
                    * a complex query method that combines (almost) all of the above (and more): `.predict()`
            There is some other methods that change the 'default' behaviour of the model, but do not actually change
            the model. See secition 'Default Values and Default Subsets'.

        Storing / Loading:

            An instance of a model can be loaded from and stored in a file using `Model.load()` and `Model.store()`.

    Implementing a concrete subclass of the Model class:

        The abstract `Model` class defines stubs for those methods that actual models are required to implement. The
        stubs are:

          * _set_data(self, df, drop_silently, **kwargs)
          * _fit(self)
          * _marginalizeout(self, keep, remove)
          * _conditionout(self, keep, remove)
          * _density(self, x)
          * _sample(self)
          * copy(self, name=None)

        Each stub also contains documentation that provides information about guarantees (i.e. properties that hold
        when the stub's implementation is called) and responsibilities (i.e. properties and tasks that the stub has
        to fulfill). It furthermore gives hints for implementation because often existing methods and functions may
        be reused.

        Each concrete subclass must also fill the `._aggrMethods` dictionary. See the section 'Private
        Attributes:_aggrMethods'.

        Each class may additionally implement a number of other methods:

          * _probability()
          * _generate_model()

     Private Attributes:

           _aggrMethods : dict

                This dictionary maps a string identifier of an aggregation method to the actual method implementation,
                i.e. a class method that returns the desired aggregation. See for example `Gaussian.py`, and there the
                 __init__ method where `.aggrMethods` is assigned the method `.maximum` as well as the implementation
                 of that method which simply returns the mean of the Gaussian.

                The possible aggregation methods are listed in `base.AggregationMethods`. Of these, 'density' and
                'probability' are treated elsewhere (i.e. by `._density()` and `._probability()` and your model
                 implementation must not supply its own implementation of it.
                 Hence, in your class you should set class methods to the keys 'maximum' and 'average'.

    """

    def __str__(self):
        """Return a string representation."""
        # TODO: add some more useful print out functions / info to that function
        return (self.__class__.__name__ + " " + self.name + "':\n" +
                "dimension: " + str(self.dim) + "\n" +
                "names: " + str([self.names]) + "\n")
        # "fields: " + str([str(field) for field in self.fields]))

    def __short_str__(self, max_fields=5):
        """Return a short string representation. By default return a string representation of the first 5 fields. Set
        the max_fields parameter to change that.
        """
        fields = self.fields[:max_fields]
        field_strs = [('#' if field.dtype == 'string' else '±') + field.name for field in fields]
        return self.name + "(" + ",".join(field_strs) + ")"

    def asindex(self, names):
        """Given a single name or a sequence of names of a field, return the indexes of these in the .field attribute of
        the model.

        Returns : int or [int]
             A single index (if a single name was given) or a sequence of indexes.
        """
        if isinstance(names, str):
            return self._name2idx[names]
        else:
            return [self._name2idx[name] for name in names]

    def byname(self, names):
        """Given a single name or a sequence of names of a field, return the corresponding single field or sequence
        of fields of this model.

        Returns : Field or [Field]
            A single `Field` (if a single name was given) or a sequence of `Field`s.
        """
        def _byname(name):
            return self.fields[self._name2idx[name]]

        if isinstance(names, str):
            return _byname(names)
        else:
            return [_byname(name) for name in names]

    def isfieldname(self, names):
        """Returns true iff the single string or list of strings given as variables names are (all) names of random
        variables of this model.
        """
        if isinstance(names, str):
            return names in self._name2idx
        else:
            return all([name in self._name2idx for name in names])

    def inverse_names(self, names, sorted_=False):
        """Given a sequence of names of fields (or a single name), returns a sorted list of all names
        of fields in this model which are _not_ in names. The order of names is the same as the order of
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
        """Given a set, sequence or list of fields of this model, returns a list of the
        same names but in the same order as the fields in the model.
        It silently drops any duplicate names.
        """
        return utils.sort_filter_list(names, self.names)

    def __init__(self, name="model"):
        """Construct and return a new model.

        Args:
            name [string]: name of the model. Defaults to 'model'.
        """
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
        self.history = {}
        self.parallel_processing = True

    def _setempty(self):
        self._update_remove_fields()
        return self

    def _isempty(self):
        return self.dim == 0

    def json_fields(self):
        """Returns a json-like representation of the fields of this model."""
        json_ = list(map(field_tojson, self.fields))
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

        if callbacks is not None:
            [c() for c in callbacks]

        return self

    def _set_model_params(self, **kwargs):
        """Sets the parameters of a model. See also model.set_model_params.

        This method must be implementedby all actual model classes. Make sure it complies to the requirements, see
        model.set_model_params.
        """
        raise NotImplementedError("You have to implement the _set_model_params method in your model!")

    def set_data(self, df, silently_drop=False, **kwargs):
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
        default_opts = {
            'pci_graph': False,
        }
        valid_opts = {
            'pci_graph': [True, False]
        }
        kwargs = utils.update_opts(kwargs, default_opts, valid_opts)

        # general clean up
        df = data_import_utils.clean_dataframe(df)

        if df.shape[0] == 0:
            raise ValueError("Cannot fit to data frame with no rows.")
        if df.shape[1] == 0:
            raise ValueError("Cannot fit to data frame with no columns.")

        # model specific clean up, setting of data, models fields, and possible more model specific stuff
        callbacks = self._set_data(df, silently_drop, **kwargs)

        self.mode = 'data'
        self._update_all_field_derivatives()
        if callbacks is not None:
            [c() for c in callbacks]

        self.pci_graph = pci_graph.create(self.data) if kwargs['pci_graph'] else None
        return self

    def _init_history(self):
        """Initializes the history functionality. """
        for f in self.fields:
            self.history[f['name']] = {'conditioned': [], 'marginalized': None}

    def _set_data(self, df, silently_drop, **kwargs):
        """Set the data for this model instance using the data frame `df`. After completion of this method
        the following attributes are set:

         * self.data
         * self.fields
         * self.mode
         * self._history (initialized)

        See also `.set_data()`.

        This method must be implemented by all actual model classes. For this purpose you might just want to use one
        of the following:
         * `._set_data_mixed()`: for continuous and categorical data
         * `._set_data_continuous()`: for continuous data
         * `._set_data_categorical()`: for categorical data
        """
        raise NotImplementedError("your model must implement this method")

    def _set_data_mixed(self, df, silently_drop, num_names=None, cat_names=None, **kwargs):
        """See Model._set_data"""

        # split in categorical and numeric columns
        if num_names is None or cat_names is None:
            _, cat_names, num_names = data_import_utils.get_columns_by_dtype(df)
        self._categoricals = cat_names
        self._numericals = num_names

        # check if dtype are ok
        #  ... nothing to do here ...

        # set data for model as needed
        #  reorder data frame such that categorical columns are first
        df = df[self._categoricals + self._numericals]

        # shuffle and set data frame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_data, self.data = data_import_utils.split_training_test_data(df)

        # derive and set fields
        self.fields = data_import_utils.get_discrete_fields(df, self._categoricals) + \
                      data_import_utils.get_numerical_fields(df, self._numericals)

        # normalize numerical part of data frame. For better numerical performance.
        # this is done at model level
        # if normalize:
        #        ) = normalize_dataframe(df.loc[:, self._numericals])
        #        self.data_norm =

        self._init_history()

        return self

    def _set_data_continuous(self, df, silently_drop, **kwargs):
        """see Model._set_data"""

        # split in categorical and numeric columns
        all, categoricals, numericals = data_import_utils.get_columns_by_dtype(df)

        # check if dtype are ok
        if len(categoricals) > 0:
            raise ValueError('Data frame contains categorical columns: ' + str(categoricals))

        # shuffle and set data frame
        df = pd.DataFrame(df, columns=numericals).sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_data, self.data = data_import_utils.split_training_test_data(df)

        # derive and set fields
        self.fields = data_import_utils.get_numerical_fields(self.data, numericals)

        self._init_history()

        return self

    def _set_data_categorical(self, df, silently_drop, **kwargs):
        """see Model._set_data"""

        # split in categorical and numeric columns
        all, categoricals, numericals = data_import_utils.get_columns_by_dtype(df)

        # check if dtype are ok
        if len(numericals) > 0:
            raise ValueError('Data frame contains numerical columns: ' + str(numericals))

        # shuffle and set data frame
        df = pd.DataFrame(df, columns=categoricals).sample(frac=1, random_state=42).reset_index(drop=True)
        self.test_data, self.data = data_import_utils.split_training_test_data(df)

        # derive and set fields
        self.fields = data_import_utils.get_discrete_fields(self.data, categoricals)

        self._init_history()

        return self

    def fit(self, df=None, auto_extend=False, **kwargs):
        """Fit the model. The model is fit:
        * to the optionally passed DataFrame `df`,
        * or to the previously set data (using `.set_data()`)

        On return of this method the attribute `.data` is filled with the appropriate data that
        was used to fit the model, i.e. if `df` is given it is set using `.set_data()`

        Args:fit(
            df: pd.DataFrame, optional
                The pandas data frame that holds the data to fit the model to. You can also
                previously set the data to fit to using the set_data method.

        Returns:
            The modified, fitted model.
        """
        if df is not None:
            return self.set_data(df, **kwargs).fit(**kwargs)

        if self.data is None and self.mode != 'data':
            raise ValueError(
                'No data frame to fit to present: pass it as an argument or set it before using set_data(df)')

        try:
            callbacks = self._fit(**kwargs)

            self.mode = "both"
            # self._update_all_field_derivatives()
            if callbacks is not None:
                [c() for c in callbacks]

            if auto_extend:
                auto_extent.adopt_all_extents(self)

        except NameError:
            raise NotImplementedError("You have to implement the _fit method in your model!")
        return self

    def marginalize(self, keep=None, remove=None):
        """Marginalize fields out of the model. Either specify which fields to keep or specify
        which to remove.

        Note that marginalization depends on the domain of a field. That is: if its domain is bounded it is
        conditioned on this value (and marginalized out). Otherwise it is 'normally' marginalized out (assuming that
        the full domain is available).

        Hidden fields:
          * You may marginalize hidden fields.
          * hidden fields are always kept, unless they are explicitly included in the argument <remove`.

        Default values / default subsets:
            If a field that has a default value is to be marginalized out, it is instead conditioned out on its
            default.

        Arguments:

            keep: string or a sequence of strings, or field of sequence of fields
                Names of fields or fields of a model. The given fields of this model are kept. All other random
                variables are marginalized out. You may specify the special string "*", which stands for 'keep all
                fields'

            remove: string or sequence of string, or field or sequence of fields
                Names of fields or fields of a model. The given fields of this model
                are marginalized out.

        Returns:
            The modified model.
        """
        # logger.debug('marginalizing: ' + ('keep = ' + str(keep) if remove is None else ', remove = ' + str(remove)))

        if keep is not None and remove is not None:
            raise ValueError("You may only specify either 'keep' or 'remove', but not both.")

        if keep is not None:
            if keep == '*':
                return self
            keep = base.to_name_sequence(keep)
            if self._hidden_count != 0:
                keep = keep + [f['name'] for f in self.fields if f['hidden']]  # keep hidden dims
            if not self.isfieldname(keep):
                raise ValueError("invalid field names in argument 'keep': " + str(keep))
        elif remove is not None:
            remove = base.to_name_sequence(remove)
            if not self.isfieldname(remove):
                raise ValueError("invalid field names in argument 'remove': " + str(remove))
            keep = set(self.names) - set(remove)
        else:
            raise ValueError("Missing required argument: You must specify either 'keep' or 'remove'.")

        if len(keep) == self.dim or self._isempty():
            return self
        if len(keep) == 0:
            return self._setempty()

        keep = self.sorted_names(keep)
        remove = self.inverse_names(keep, sorted_=True)

        # condition out any fields with default values/subsets on that very default
        conditions = []
        defaulting_dims = []
        for name in remove:
            field = self.byname(name)
            if field['default_subset'] is not None:
                c = Condition(name, "in", field['default_subset'])
            elif field['default_value'] is not None:
                c = Condition(name, "==", field['default_value'])
            else:
                continue
            defaulting_dims.append(name)
            conditions.append(c)
        if len(conditions) > 0:
            self.condition(conditions)._marginalize(remove=defaulting_dims)

        return self._marginalize(keep)

    def _marginalize(self, keep=None, remove=None):
        """Marginalize the fields given in remove, keeping those in keep.

        This is the internal version of the public method marginalize. You probably want to call that one, not _marginalize.

        Args:
            keep: the names of the fields to keep, in the same order than in self.fields.
            remove: the names of the fields to remove, in the same order than in self.fields.

        Notes:
            You have to specify at least one of keep and remove.
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
                # need to call this, since it will not be called later in this particular case
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
            if callbacks is not None:
                [c() for c in callbacks]
            remove = self.inverse_names(keep, sorted_=True)  # do it again, because fields may have changed
            for name in cond_out:
                self.history[name]['marginalized'] = 'conditioned_out'

        if len(keep) != self.dim and not self._isempty():
            callbacks = self._marginalizeout(keep, remove)

            self._update_remove_fields(remove)
            if callbacks is not None:
                [c() for c in callbacks]
            for name in remove:
                self.history[name]['marginalized'] = 'marginalized_out'
        return self

    def _marginalizeout(self, keep, remove):
        """Marginalize the model such that only fields with names in `keep` remain and
        fields with name in `remove` are removed.

        This method must be implemented by any actual model that derived from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
          * keep is anything but a list of names of fields of this model
          * keep is empty
          * the model itself is empty
          * keep contains all names of the model
          * remove is anything but the inverse list of names of fields with respect to keep

        Moreover if `._marginalizeout()`is called, keep and remove are guaranteed to be in the same order than the
        fields of this model
        """
        raise NotImplementedError("Implement this method in your model!")

    def condition(self, conditions=None, is_pure=False):
        """Condition this model according to the list of three-tuples (<name-of-random-variable>, <operator>,
        <value(s)>). In particular objects of type ConditionTuples are accepted and see there for allowed values.

        Note: This only restricts the domains of the fields. To remove the conditioned field you need to call
        marginalize with the appropriate parameters.

        TODO: this is actually a conceptual error. It should NOT only restrict the domain, but actually compute a
          conditional model. See  https://ci.inf-i2.uni-jena.de/gemod/modelbase/issues/4

        Hidden fields: Whether or not any field of the model is hidden makes no difference to the result of this
        method. In particular you may condition on hidden fields.

        Default values: Default values for fields are not taken into consideration in any way.

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

        # condition the domain of the fields
        names = []
        for (name, operator, values) in conditions:
            self.byname(name)['domain'].apply(operator, values)
            names.append(name)
            # store history
            e = {'operator': operator, 'value': values}
            self.history[name]['conditioned'].append(e)

        # condition model
        # todo: currently, conditioning of a model is always only done when it is conditioned out.
        # in the future this should be decided by the model itself:
        #  for an emperical model it is always possible to compute the conditional model
        #  for an gaussian model we may compute it for point conditinals right away, but we maybe would not want to do it for range conditionals
        # TODO: if conditions is a zip: how can it be reused a 2nd and 3rd time below!??
        # condition data
        if self.mode == 'data' or self.mode == 'both':
            self.data = data_operations.condition_data(self.data, conditions)
            self.test_data = data_operations.condition_data(self.test_data, conditions)

        self._update_extents(names)
        return self

    def _conditionout(self, keep, remove):
        """Condition the field with name in `remove` on their available, //not unbounded// domain and
        marginalizes them out.

        This method must be implemented by any actual model that derived from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
          * remove is anything but a list of names of fields of this model
          * remove is empty
          * the model itself is empty
          * remove contains all names of the model
          * keep is anything but the inverse list of names of fields with respect to remove

        Moreover keep and remove are guaranteed to be in the same order than the fields of this model

        Note that we don't know yet how to condition on a non-singular domain (i.e. condition on interval or sets).
        As a work around we therefore:
          * for continuous domains: condition on (high-low)/2
          * for discrete domains: condition on the first element in the domain
        """
        raise NotImplementedError("Implement this method in your model!")

    def hide(self, dims, val=True):
        """Hides/unhides the given fields.

        Hiding a field does not cause any actual change in the model. It simply removes the hidden field from
        any output generated by the model. In combination with setting a default value it provides you with a view on
        a slice of the model, where hidden fields are fixed to their default values / subsets.

        Notes:
          * fields without defaults values/subsets can be hidden anyway.
          * any hidden field is still reported as a field of the model by means of auxiliary methods like
          `Model.byname`, `Model.isfieldname`, `Model.fields`, etc.

        However, hiding a field with name 'foo' has the following effects on subsequent queries against the model:
          * density: If no value for 'foo' is specified, the default value (which must be set) is used and the density
            at that point is returned.
          * probability: If no subset for 'foo' is specified, the default subset (which must be set) is used and the
            density at that point is returned
          * predict: Predictions are done on the conditional model on the defaults. The returned tuples do not contain
            values for those fields that are both, hidden and defaulting.
          * sample: Sampling is done 'normally', but the hidden fields are removed from the generated samples.
          * condition: no change in behaviour what so ever. In particular, it is possible to condition hidden
            fields.
          * marginalize: no change in behaviour either. However, if there is a default value set, the behaviour changes.
            See set_default_* for more.

        For more details, look at the documentation of each of the methods.

        Args:
            dims:
               * a single field, or
               * a single field name, or
               * a sequence of the above, or
               * a dict of <name>:<bool>, where <name> is the name of the fields to hide (<bool> == True) or unhide
                (<bool> == False)

            val (optional, defaults to True)
               * if a dict is provided for dims, this is ignored, otherwise
               * set to True to hide, or False to unhide all given fields.
        """

        if dims is None:
            dims = {}

        # normalize to dict
        if not isinstance(dims, dict):
            dims = base.to_name_sequence(dims)
            dims = dict(zip(dims, [val] * len(dims)))

        for name, flag in dims.items():
            field = self.byname(name)
            self._hidden_count -= field['hidden'] - flag
            field['hidden'] = flag
        return self

    def hidden_fields(self, invert=False):
        """Return the sequence of names of fields that are hidden. Their rder matches the order of fields in the
        model.
        """
        return [f['name'] for f in self.fields if (invert - f['hidden'])]

    def hidden_idxs(self, invert=False):
        """Return the sequence of indexes to fields that are hidden in increasing order.
        """
        return [idx for idx, f in enumerate(self.fields) if (invert - f['hidden'])]

    def reveal(self, dims):
        """Reveals the given fields. Provide either a single field or a field name or a sequence of these.

        See also `Model.hide()`.
        """
        return self.hide(dims, False)

    def set_default_value(self, dims, values=None):
        """Sets default values for fields.

        There is two ways of specifying the arguments:

        1. Provide a dict of <field-name> to <default-value> (as first argument or argument dims).
        2. Provide two sequences: dims and values that contain the field names and default values, respectively.

        Note that setting defaults is independent of hiding a field.

        The abstract idea of default values is as follows:

        Once a default value is set, any operation that returns or uses values of that field will have or use
        that particular value, respectively. See the documentation of the interface methods to understand the
        detailed implications.

        Returns self.
        """

        if values is None:
            if dims is None:
                dims = {}
            # dims is a dict of <field-name> to <default-value>
            if not isinstance(dims, dict):
                raise TypeError("dims must be a dict of <field-name> to <default-value>, if values is None.")
        else:
            # dims and values are two lists
            if len(dims) != len(values):
                raise ValueError("dims and values must be of equal length.")
            dims = dict(zip(dims, values))

        if not self.isfieldname(dims.keys()):
            raise ValueError("fields must be specified by their name and must be a field of this model")

        # update default values
        for name, value in dims.items():
            field = self.byname(name)
            if not field['domain'].contains(value):
                raise ValueError("The value to set as default must be within the domain of the field.")
            field['default_value'] = value

        return self

    def set_default_subset(self, dims, subsets=None):
        """Sets default subsets for fields.

        There is two ways of specifying the arguments:

        1. Provide a dict of <field-name> to <default-subset> (as first argument or argument dims).
        2. Provide two sequences: dims and subsets that contain the field names and default subsets, respectively.

        Note that setting defaults  is independent of hiding a field.

        The default subset of a field is used when any probability over that field is queried, but no value
        for the field is provided. Then the default subset value is used instead. See the documentation of the
        interface methods to understand the detailed implications.

        Returns self.
        """
        if subsets is None:
            if dims is None:
                dims = {}
            # dims is a dict of <field-name> to <default-value>
            if not isinstance(dims, dict):
                raise TypeError("dims must be a dict of <field-name> to <default-value>, if values is None.")
        else:
            # dims and values are two lists
            if len(dims) != len(subsets):
                raise ValueError("dims and values must be of equal length.")
            dims = dict(zip(dims, subsets))

        if not self.isfieldname(dims.keys()):
            raise ValueError("fields must be specified by their name and must be a field of this model")

        # update default values
        for name, subset in dims.items():
            field = self.byname(name)
            if not field['domain'].contains(subset):
                raise ValueError("The value to set as default must be within the domain of the field.")
            field['default_subset'] = subset

        return self

    @staticmethod
    def has_default(dims):
        """Given a single field name or a sequence of field names returns a single or a sequence of booleans,
        each indicating whether or not the field with that name has any defaulting value/subset.
        """

        def _has_default(dim):
            return dim['default_value'] is not None or dim['default_subset'] is not None

        if isinstance(dims, str):
            return _has_default(dims)
        else:
            return map(_has_default, dims)

    def aggregate_data(self, method, opts=None):
        """Aggregate the models data according to given method and options, and return this aggregation.

        See `data_aggregation.aggregate_data()`.
        """
        # TODO: should I add the option to compute aggregations on a selectable part of the data
        return data_aggregation.aggregate_data(self.data, method, opts)

    def aggregate_model(self, method, opts=None):
        """Aggregate the model according to given method and options and return the aggregation value.

        Args:
            method: string
                The method how to aggregate. It depends on the model class which methods are available.
            opts: {}
                Not used.
        """

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
        """ Aggregate the model using the given method and return the aggregation as a list. The order of elements
        in the list matches the order of fields in fields attribute of the model.

        Args:
            method: string
                The method how to aggregate. It depends on the model class which methods are available.
            opts: Any, optional.
                Additional arguments for the aggregation.

        Hidden fields:
            Any value of a field that is hidden will be removed from the resulting aggregation before returning.

        Default values:
            Instead of the aggregation of the model, the aggregation of the conditional model on all defaulting
            fields is returned. For example:

                Let p(sex,age) be a bivariate model and let its maximum be at argmax(p(sex,age)) = ('Female',
                18). However, if the default of field 'sex' is set to 'Male' the aggregation of the model is (
                'Male', argmax(p(age|sex='Male')) instead. If could be, for example, ('Male', 23).

        TODO: right now default values are not taken into consideration in this way...

        Returns:
            The aggregation of the model as a sequence.
        """
        if self._isempty():
            raise ValueError('Cannot aggregate a 0-dimensional model')

        # derive conditional model for default values. prefers default_subset over default_value
        model = self
        dims_subset_default = {}  # TODO implement this
        dims_value_default = {}

        if len(dims_subset_default) > 0 or len(dims_value_default) > 0:
            conditions_value = [(k, "==", v) for k, v in dims_value_default.items()]
            conditions_subset = [(k, "in", v) for k, v in dims_subset_default.items()]

            conditions = conditions_value
            conditions.update(conditions_subset)

            model = self._cond_default_model = self.copy().model(where=conditions)
        # TODO: CONTINUE HERE. make use of "model"!

        model_res = self.aggregate_model(method, opts)

        # TODO: add values of fields that defaulted to some value

        # remove values of hidden fields
        # TODO: move to own method
        if self._hidden_count != 0:
            idxs = self.hidden_idxs(invert=True)  # returns the non-hidden indexes
            model_res = [model_res[i] for i in idxs]

        return model_res

    def density(self, values=None, names=None):
        """Returns the density at given point.

         Args:
            There are several ways to specify the arguments:

            (1) A list of values in `values` and a list of fields names in `names`, with corresponding order.
            They need to be in the same order than the fields of this model.

            (2) A list of values in `values` in the same order than the fields of this model. `names` must not be
            passed. This is faster than (1).

            (3) A dict of key=name:value=value in `values`. `names` is ignored.

        Hidden fields and default values:

            The density will be computed on the model over all (i.e. hidden and non-hidden) fields. If you do not
            specify a value for a field (hidden or not) its default value will be used instead to compute the
            density at that point.

            If you use variant (1) and (3) of specifying arguments: You may or not provide a value for fields
            with default values. You may also 'mix', i.e. give values for some fields that have default values,
            but not for other fields that also have default values.

            If you use variant (2): You must not specify the value of any hidden field. And you must specify
            values for all non-hidden fields. This is because it would be impossible to determine what field they
            belong to.

        Internal:

            It may happen that some elements of values are not scalars such as 1 or 'foo' but a single item list such
            as [1] or ['foo']. This is because the split-method 'elements' is used to create input for both:

              * probability (requiring a sequence of domains), and
              * density (requiring a sequence of scalars)

            Currently, we only allow scalar values at input and instead we take care of it in `Model.predict()`.
        """
        if self._isempty():
            raise ValueError('Cannot query density of 0-dimensional model')

        # normalize parameters to ordered list form
        if isinstance(values, dict):
            values = [values.get(name, self.byname(name)['default_value']) for name in self.names]
        elif names is not None and values is not None:
            # unordered list was passed in
            if len(values) != len(names):
                raise ValueError('Length of names and values does not match.')
            elif len(set(names)) == len(names):
                raise ValueError('Some name occurs twice.')
            values = dict(zip(values, names))  # to dict
            values = [values.get(name, self.byname(name)['default_value']) for name in self.names]  # to
        elif len(values) == (self.dim - self._hidden_count):
            # correctly ordered list was passed in
            if self._hidden_count != 0:
                # merge with defaults of hidden fields (which may not have been passed!)
                i = iter(values)
                values = [f['default_value'] if f['hidden'] else next(i) for f in self.fields]
            else:
                pass  # nothing to do]
        else:
            raise ValueError("Invalid number of values passed.")

        p = self._density(values)
        return p

    def _density(self, x):
        """Return the density of the model at point `x`.

        This method must be implemented by any actual model that derives from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:

          * `x` is anything but a list of values as input for density
          * `x` has a length different than the dimension of the model
          * the model itself is empty

        It is _not_ guaranteed, however, that the elements of `x` are of matching type / value for the model

        Args:
            x: sequence
                List of values as input for the density.
        """
        raise NotImplementedError("Implement this method in your model!")

    def probability(self, domains=None, names=None):
        """
        Return the probability of given event.

        By default this returns an approximation to the true probability. To implement it exactly or a different
        approximation for a model class re-implement the method `_probability()`. See `Model._probability()` for
        detail on the generic implementation.

        Args:
            There are several ways to specify the arguments:

            (1) A list of domains in `domains` and a list of fields names in `names`, with corresponding order.
            They need to be in the same order than the fields of this model.

            (2) A list of domains in `domains` in the same order than the fields of this model. `names` must not
            be passed. This is faster than (1).

            (3) A dict of key=name:value=domain in `domains`. `names` is ignored.

        Hidden fields and default values:

            See Model.density().

        """
        if self._isempty():
            raise ValueError('Cannot query density of 0-dimensional model')

        # normalize parameters to ordered list form
        if isinstance(domains, dict):
            # dict was passed
            domains = [domains.get(name, self.byname(name)['default_subset']) for name in self.names]
        elif names is not None and domains is not None:
            if len(domains) != len(names):
                raise ValueError('Length of names and values does not match.')
            elif len(set(names)) == len(names):
                raise ValueError('Some name occurs twice.')
            domains = dict(zip(domains, names))  # to dict
            domains = [domains.get(name, self.byname(name)['default_subset']) for name in self.names]  # to domains
        elif len(domains) == (self.dim - self._hidden_count):
            # correctly ordered list was passed in
            if self._hidden_count != 0:
                # merge with defaults of hidden fields (which may not have been passed!)
                i = iter(domains)
                domains = [f['default_domain'] if f['hidden'] else next(i) for f in self.fields]
            else:
                pass  # nothing to do
        else:
            raise ValueError("Invalid number of values passed.")

        # model probability
        model_prob = self._probability(domains)
        return model_prob

    def _probability(self, domains):
        try:
            cat_len = len(self._categoricals)
        except AttributeError:
            # self._categoricals might not be implemented in a particular model. this is the fallback:
            cat_len = sum(f['dtype'] == 'string' for f in self.fields)
        return self._probability_generic_mixed(domains[:cat_len], domains[cat_len:])

    def _probability_generic_mixed(self, cat_domains, num_domains):
        """
        Returns an approximation to the probability of the given event.

        This works generically for any model mixed model that stores categorical fields before numerical
        fields. It is assumed that all domains in `cat_domains` and `num_domains` are given in their respective
        order in the model.

        Args:
            cat_domains: sequence of domains
                A domain describes the subspace that the event covers for a particular fields.
                A valid domain categorical domain is a sequence of strings, e.g. ['A'], or ['A','B']
            num_domains: sequence of domains
                A domain describes the subspace that the event covers for a particular fields.
                A valid domain quantitative domain is a 2-element sequence or tuple, e.g. [1,2], or [1,1] or [2,6].
                If [l,h] is the interval, then neither l nor h may be +-infinity and it must hold l <= h
        """
        # volume of all combined each quantitative domains
        vol = functools.reduce(operator.mul, [high - low for low, high in num_domains], 1)
        # map quantitative domains to their mid
        y = [(high + low) / 2 for low, high in num_domains]
        #y = [int((high + low) / 2) for low, high in num_domains]

        # sum up density over all elements of the cartesian product of the categorical part of the event
        # TODO: generalize
        assert (all(len(d) == 1 for d in cat_domains)), "did not implement the case where categorical domain has more than one element"
        x = list([d[0] for d in cat_domains])
        return vol * self._density(x + y)
        #return vol * self._density(list(cat_domains) + y)

    def sample(self, n=1):
        """Returns n samples drawn from the model as a dataframe with suitable column names.
        TODO: make interface similar to select_data
        """
        if self._isempty():
            raise ValueError('Cannot sample from 0-dimensional model')
        samples = (self._sample() for i in range(n))
        hidden_dims = [f['name'] for f in self.fields if f['hidden']]
        return pd.DataFrame.from_records(data=samples, columns=self.names, exclude=hidden_dims)

    def _sample(self):
        """Returns a single sample drawn from the model.

        This method must be implemented by any actual model that derives from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
            * the model is empty
        """
        raise NotImplementedError()

    def copy(self, name=None):
        """Returns a copy of this model, optionally using a new name.

        This method must be implemented by any actual model that derives from the abstract model class.

        Note: use `Model._defaultcopy()` to get a copy of the 'abstract part' of an model
        instance and then only add your custom copy code.
        """
        raise NotImplementedError()

    def _defaultcopy(self, name=None):
        """Return a new model of the same type with all instance variables of the abstract base model copied:
          * data (a reference to it!!!)
          * fields (deep copy)
        """
        name = self.name if name is None else name
        mycopy = self.__class__(name)
        mycopy.data = self.data  # .copy()
        mycopy.test_data = self.test_data.copy()
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy.mode = self.mode
        mycopy._update_all_field_derivatives()
        mycopy.history = cp.deepcopy(self.history)
        mycopy.parallel_processing = self.parallel_processing
        return mycopy

    def _condition_values(self, names=None, pairflag=False, to_scalar=True):
        """Return the list of values to condition on given a sequence of field names to condition on.

        Essentially, this is a look up in the domain restrictions of the fields to be conditioned on (i.e. the
        sequence names).

        Args:
            names: sequence of strings
                The field names to get the conditioning domain for. If not given the condition values for all
                fields are returned.
            pairflag: bool, optional.
                Defaults to `False`. If set to `True` not a list of values but a zip-object of the names and the
                values to condition on is returned
            to_scalar: bool
                Flag to turn any non-scalar values to scalars. Defaults to `False`.
        """
        if not to_scalar:
            raise NotImplemented
        fields = self.fields if names is None else self.byname(names)
        # It's not obvious but this is equivalent to the more detailed code below...
        cond_values = [field['domain'].mid() for field in fields]

        # cond_values = []
        # for field in fields:
        #     domain = field['domain']
        #     dvalue = domain.value()
        #     if not domain.isbounded():
        #         raise ValueError("cannot condition fields with not bounded domain!")
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
        """Store the model to a file at `filename`.

        You can load a stored model using `Model.load()`.
        """
        with open(filename, 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """Load the model in file at `filename`.

        You can store a stored model using `Model.store()`.
        """
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
        """Updates `Model.extents` of the fields with names in the sequence `to_update`.
        Updates all if `to_update` is None.

        `Model.extents` is only a convenience lookup dictionary, hence it needs be updated if the actual
        extents change for whatever reason.
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
        """Updates (i.e. recreates) the `Model._name2idx` dictionary from current `self.fields`.

        This is necessary whenever `Model.fields` change.
        """
        self._name2idx = dict(zip([f['name'] for f in self.fields], range(self.dim)))

    def _update_remove_fields(self, to_remove=None):
        """Remove the fields in the sequence `to_remove` (and correspondingly derived structures) from `self.fields`.
         Removes all if `to_remove` is None.
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
        """Rebuild all field derivatives. `self.fields` must be set before this method is called.
        @fields

        field derivatives include:
         * `Model.dim`
         * `Model._name2idx`
         * `Model.names`
         * `Model.extents`
         * `Model._hidden_count`
        """
        self.dim = len(self.fields)
        self._update_name2idx_dict()
        self.names = [f['name'] for f in self.fields]
        self._update_extents()
        self._hidden_count = sum(map(lambda f: f['hidden'], self.fields))
        return self

    def model(self, model='*', where=None, as_=None, default_values=None, default_subsets=None, hide=None):
        """Return a model with name `as_` that models the fields in `model` respecting conditions in
        `where`. Moreover, fields in hide will be hidden/unhidden (depending on the passed boolean) and default
        values and/or default subsets will be set.

        It does NOT create a copy, but MODIFIES this model.

        Args:
            model: sequence of strings or "*", optional.
                Each string is the name of a field to model. Its value may also be "*" or ["*"], meaning all random
                variables of this model. Defaults to '*'.
            where: sequence of tuples, optional.
                The sequence of conditions to model. Defaults to None.
            as_: string, optional
                The name for the model to derive. If set to None the name of the base model is used. Defaults to None.
            default_values: dict, optional
                 A dict of <name>:<value>, where <name> is the name of a field of this model and <value> the
                 default value to be set. Pass None to remove a already set default.
            default_subsets: dict, optional.
                An dict of <name>:<subset>, where <name> is the name of a field of this model and <subset> the
                default subset to be set. Pass None to remove the default.
            hide: string or sequence of strings, optional.
                Each name refers to a field of this model that will be hidden. Or a <name>:<bool>-dict where name is
                a fields name and bool is True iff the field will be hidden, or False if it is unhidden.

        Returns:
            The modified model.
        """
        self.name = self.name if as_ is None else as_

        return self.set_default_value(default_values) \
            .set_default_subset(default_subsets) \
            .hide(hide).condition(where) \
            .marginalize(keep=model)

    def predict(self, predict, where=None, splitby=None, for_data=None, returnbasemodel=False):
        """Calculate the prediction against the model and returns its result as a pd.DataFrame.

        The data frame contains exactly those fields which are specified in 'predict'. Its order is preserved.

        It does NOT modify the model it is called on.

        Args:
            predict: list
                A list of 'things' to predict. Each of these 'things' may be: a `Field`, a str identifying a `Field`,
                or an 'AggregationTuple'.
                These 'things' are included in the returned data frame, in the same order as specified here.

            where: list[ConditionTuple], optional.
                A list of `ConditionTuple`s, representing the conditions to adhere.

            splitby: list[SplitTuple], optional.
                A list of 'SplitTuple's, i.e. a list of fields on which to split the model and the method how to
                do the split.

            for_data: pd.DataFrame, optional.
                Set of data points to do prediction for. Is combined with the values of `splitby` (if they overlap).
                The columns of the dataframe must be labelled with the corresponding names of the dimensions in self.

            returnbasemodel: bool
                If set this method will return the pair (result-dataframe, basemodel-for-the-prediction). Defaults to
                False.

        Returns:

            A `pd.DataFrame` with the fields as given in 'predict', or a tuple (see returnbasemodel).

        Evidence and splits:

            While both may be used in parallel, they may currently not share any dimensions.

        Hidden fields:

            You may not include any hidden field in the predict-clause and such queries will result in a
            ValueError().
            TODO: fix?
            TODO: raise error

        Default Values:

            You may leave out splits for fields that have a default value, like this:

                # let model be a model with the default value of X set to 1.
                model.predict(Density('X','Y'), splitby=Split('Y', 'equidist, 10))

            This will NOT result in an error, since the - normally required - split for X is automatically generated,
            due to the information that X defaults to 1.

            You may, at the same time, include the defaulting fields in the result table. A slight variation of
            the example above:

                # let model be a model with the default value of X set to 1.
                model.predict(X, Y, Density('X','Y'), splitby=Split('Y', 'equidist, 10))

        for_data-clause:

            You may provide data that a prediction is done for. See 'input generation'.

        Splits:

            Splits may have scalars or domains as a result type. See 'input generation'.

        Input generation:

            Many queries require 'input' that a query is computed on. E.g. a density can only be computed at a given
            point. There is two ways to provide input: using splits (`SplitTuple`) and with the `for_data`-clause.

            In case of the `for_data` clause the data points are taken as they are and the query is computed on them.
            E.g.:

            > model.predict(['RW', Density([RW])], for_data=pd.DataFrame(data={'RW': [5, 10, 15]}))
            >
            >   RW  density(['RW'])
            >   5         0.005603
            >  10         0.248187
            >  15         0.250418

            In the case of splits, each split is 'expanded' into a series of values of its dimension. Then these
            splits are combined as their cartesian product, i.e. a cross-join is done. This allows to create all
            sorts of 'meshes' across dimensions. For example:

            > model.predict(['sex', 'FL', Probability([FL, sex])], splitby=[Split(sex), Split(FL)])
            >
            >        sex         FL  @probability(['FL', 'sex'])
            >   0   Female   3.450144                     0.000545
            >   1   Female   4.472832                     0.001566
            > ...
            >   24  Female  27.994656                     0.000781
            >   25    Male   3.450144                     0.000633
            >   26    Male   4.472832                     0.001634
            >   27    Male   5.495520                     0.003874
            >   28    Male   6.518208                     0.008429
            > ...
            >   49    Male  27.994656                     0.000844

            Both, splits and for_data may used at the same time. They are combined as their cross product.


        Ideas for Improvement:

           How to efficiently query the model? how can I vectorize it? I believe that depends on the query. A typical
           query consists of dimensions for splits and then aggregations and densities. For the case of aggregations
           a conditioned model has to be calculated for every split. I don't see how to vectorize / speed this up
           easily. For densities it might be very well possible, as the splits are now simply input to some density
           function. It might actually be faster to first condition the model on the dimensions (values)
           and then derive the measure models... note: for density however, no conditioning on the input is required

        TODO:
            * just an idea: couldn't I merge the partial data given (takes higher priority) with all default of all
           variables and use this as a starting point for the input frame??
            * can't we handle the the split-method 'data' (which uses .test_data as the result of the split) as partial
             data. This seem more clean and versatile.

        """
        partial_data = for_data
        if partial_data is None:
            partial_data = pd.DataFrame()
        elif not self.isfieldname(partial_data.columns):
            raise ValueError('partial_data contains data dimensions that are not modelled by this model')

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

        # (0) create data structures for clauses
        aggrs, aggr_ids, aggr_input_names, aggr_dims, \
        predict_ids, predict_names, \
        split_names, name2split, \
        partial_data, partial_data_names\
            = models_predict.create_data_structures_for_clauses(self, predict, where, splitby, partial_data)
        # set of names of dimensions that we need values for in the input data frame
        input_names = aggr_input_names | set(split_names) | set(partial_data_names)

        # (1) derive base model, i.e. a model on all requested fields and measures respecting filters
        basenames = input_names.union(aggr_dims)
        basemodel = self.copy().model(model=basenames, where=where, as_=self.name + '_base')

        # (2) generate all input data
        partial_data, split_data = models_predict.generate_all_input(basemodel, splitby, split_names, partial_data)

        # (3) execute each aggregation
        aggr_model_id_gen = utils.linear_id_generator(prefix=self.name + "_aggr")
        result_list = []
        for aggr, aggr_id in zip(aggrs, aggr_ids):
            # derive submodel for aggr
            aggr_model = models_predict.derive_aggregation_model(basemodel, aggr, input_names, next(aggr_model_id_gen))
            aggr_method = aggr[METHOD_IDX]

            # Input Data: Generating input in a performant way is a bit involved:
            # * input should be generated on individual basis of an aggregation.
            #   * p(A,B|C,D) should in an outer loop generate the conditional models over C, D (which is expensive) and
            #     then on an inner loop query the actual density over A,B (which is fast).
            #   * for p(C,D|A,B) it is the inverse order
            # * for that reason each aggr generates its own input and returns it
            # * however, the order between multiple aggregations will typically not match. Hence we need to:
            #   1. also return the input data frames from each aggregation execution (and not only the output)
            #   2. and use the input information to join the multiple output data frames together

            # TODO: I imagine there is a smart way of reordering the results without having to explicitly create the
            #  cross join of cond_out_data and input_data!?

            # query model
            if aggr_method == 'density' or aggr_method == 'probability':
                aggr_df = models_predict.\
                    aggregate_density_or_probability(aggr_model, aggr, partial_data, split_data, name2split, aggr_id)
            elif aggr_method == 'maximum' or aggr_method == 'average':  # it is some aggregation
                # TODO: I believe all max/avg aggregations require the identical input data, because i always condition
                #  on all input items --> reuse it. This is: generate input before and then pass it in
                aggr_df = models_predict.\
                    aggregate_maximum_or_average(aggr_model, aggr, partial_data, split_data, name2split, aggr_id)
            else:
                raise ValueError("Invalid 'aggregation method': " + str(aggr_method))

            result_list.append(aggr_df)

        # (4) need to merge all data frames on input_names, since they are not necesarily in the same order
        # TODO: right now we do not use any indexes for merging - which probably is slower...
        #  but if we do, I think we can do this:
        #  data_frame = result_list[0].join(result_list[1:], on=input_names)

        # reduce to one final data frame
        if len(result_list) == 0:
            # no actual model query involved - simply a join of splits and partial_data
            # -> generate input (i.e. join) for requested output (i.e. predict_ids)
            partial_data_res = partial_data.loc[:, partial_data.columns & set(predict_ids)]
            split_res = (split_data[name] for name in predict_ids if name in split_data)
            dataframe = models_predict.crossjoin(*split_res, partial_data_res)
        elif len(input_names) == 0:
            # there is no index to merge on, because there was no spits or partial_data
            assert all(1 == len(res.columns) for res in result_list)
            dataframe = pd.concat(result_list, axis=1, copy=False)
        elif len(result_list) == 1:
            dataframe = result_list[0]
        else:
            dataframe = functools.reduce(lambda df1, df2: df1.merge(df2, on=list(input_names), how='inner', copy=False),
                                          result_list[1:], result_list[0])

        # (4) Fix domain valued splits.
        # Some splits result in domains (i.e. tuples, and not just single, scalar values). However,
        # I cannot currently handle intervals on the client side easily. Therefore we turn it back into scalars. Note
        # that only splits may result in tuples, but partial_data is currently not allowed to have tuples
        for name, split in name2split.items():
            if split['return_type'] == 'domain':
                dataframe[name] = dataframe[name].apply(split['down_cast_fct'])

        # (5) filter on aggregations?
        # TODO? actually there should be some easy way to do it, since now it really is SQL filtering

        # (7) get correctly ordered frame that only contain requested fields
        dataframe = dataframe[predict_ids]

        # (8) rename columns to be readable (but not unique anymore)
        dataframe.columns = predict_names

        return (dataframe, basemodel) if returnbasemodel else dataframe

    def _select_data(self, what, where=None, **kwargs):
        """Select and return that subset of the models data in `self.data` that respect the conditions in `where` and the columns
        selection in `what`.

        Args:
            what : string or sequence of strings.
            where : tuple or sequence of tuples or None
                The condition tuple to restrict the data by. Defaults to None.
            kwargs : dict, optional.
                Options dictionary. Available options are:
                    'data_category', with values:
                        'training data': return selection of training data
                        'test data': return selection of test data
                        defaults to: 'training data'

        Returns : pd.DataFrame
            The selected data as a data frame.
        """

        # todo: use update_opts also at other places where appropiate (search for validate_opts)
        opts = utils.update_opts({'data_category': 'training data'}, kwargs)
        # TODO: use validation argument for update_opts again, i.e. implement numerical ranges or such
        # opts = utils.update_opts({'data_category': 'training data'}, kwargs, {'data_category': ['training data', 'test data']})
        df = self.data if opts['data_category'] == 'training data' else self.test_data

        selected_data = data_operations.condition_data(df, where).loc[:, what]

        # limit number of returned data points if requested
        if 'data_point_limit' in kwargs:
            selected_data = selected_data.iloc[:kwargs['data_point_limit'], :]

        return selected_data

    def select(self, what, where=None, **kwargs):
        """Returns the selected attributes of all data items that satisfy the conditions as a
        pandas DataFrame.

        It selects data only, i.e. it will never return any samples from the model. To sample use sample().
        """
        # check for empty queries
        if len(what) == 0:
            return pd.DataFrame()
        if where is None:
            where = []

        # check that all columns to select are in data
        if any((label not in self.data.columns for label in what)):
            raise KeyError('at least on of ' + str(what) + ' is not a column label of the data.')

        # select data
        data = self._select_data(what, where, **kwargs)

        # return reordered
        return data.loc[:, what]

    def generate_model(self, opts={}):
        """Generate a model from no_data.

        Generate a random instance of a model class. It will replace the current model.

        This is mainly meant for testing and debugging purposes. It is not strictly required that every model class
        implements the corresponding private method `._generate_model()`.

        See the documentation of the `._generate_model()` method to learn more.
        """

        callbacks = self._generate_model(opts)  # call specific class method
        self._init_history()
        self._update_all_field_derivatives()

        if callbacks is not None:
            [c() for c in callbacks]
        return self

    def _generate_data(self, opts=None):
        """Sample `opts['n']` many samples and set it as the models data.

        When called this must be a model with `self.mode == "model or self.mode == "both"`.

        Args:
            opts : dict, optional.
                Options. Allows values are:
                    'n': int, optional.
                        The number of data items to generate. Defaults to 1000.
        """
        if opts is None:
            opts = {}
        n = opts['n'] if 'n' in opts else 1000

        if self.mode != "model" and self.mode != "both":
            raise ValueError("cannot sample from empty model, i.e. the model has not been learned/set yet.")

        self._set_data(df=self.sample(n), drop_silently=False)
        self.mode = 'both'
        return self

    def loglikelihood(self, data=None):
        """Calculate the log-likelihood of set of data points.

        Args:
            data : pd.DataFrame or object other that np.array accepts.

        """
        if data == None:
            data = self.data
        # TODO: why is there np.array ? isn't that slow?
        # TODO: why pass data in the first place?
        # TODO: why sum over it?
        return sum([np.log(self._density(x)) for x in np.array(data)])


if __name__ == '__main__':
    import pandas as pd
    import mb_modelbase as mb

    df_ = pd.read_csv('../../../mb_data/mb_data/iris/iris.csv')
    m = mb.MixableCondGaussianModel()
    m.fit(df_)
    res = m.predict(predict=['sepal_width', mb.Aggregation('sepal_length')],
                    splitby=mb.SplitTuple('sepal_width', 'data', [5]))
    print(res)

    res = m.predict(predict=['sepal_width', 'petal_width', mb.Aggregation(['sepal_length'])],
                    splitby=[mb.SplitTuple('sepal_width', 'data', [5]), mb.SplitTuple('petal_width', 'data', [5])])
    print("\n")
    print(res)
