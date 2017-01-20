# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

This module defines:

   * Model: an abstract base class for models.
   * Field: a class that represent random variables in a model.

"""
import pandas as pd
import copy as cp
from collections import namedtuple
from functools import reduce
import pickle as pickle
import logging
import splitter as sp
import utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

""" Development Notes (Philipp)
# interesting links
https://github.com/rasbt/pattern_classification/blob/master/resources/python_data_libraries.md !!!

# what about using a different syntax for the model, like the following:

    model['A'] : to select a submodel only on random variable with name 'A' // marginalize
    model['B'] = ...some domain...   // condition
    model.
"""

### GENERIC / ABSTRACT MODELS and other base classes
AggregationTuple = namedtuple('AggregationTuple', ['name', 'method', 'yields', 'args'])
SplitTuple = namedtuple('SplitTuple', ['name', 'method', 'args'])
ConditionTuple = namedtuple('ConditionTuple', ['name', 'operator', 'value'])
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
    return {'name': name, 'domain': domain, 'extent': extent, 'dtype': dtype}
""" A constructor that returns 'Field'-dicts, i.e. a dict with three components
    as passed in:
        'name': the name of the field
        'domain': the domain of the field,
        'extent': the extent of the field that will be used as a fallback for domain if domain is unbounded but a
            value for domain is required
        'dtype': the data type that the field represents. Possible values are: 'numerical' and 'string'
"""


def field_tojson(field):
    """Returns an adapted version of a field that in any case is JSON serializable. A fields domain may contain the
    value Infinity, which JSON serialized don't handle correctly.
    """
    # copy everything, but special treat domain and extent
    copy = cp.copy(field)
    copy['domain'] = field['domain'].tojson()
    copy['extent'] = field['extent'].tojson()
    return copy


def mergebyidx2(zips):
    """Merges list l1 and list l2 into one list in increasing index order, where the indices are given by idx1
    and idx2, respectively. idx1 and idx2 are expected to be sorted. No index may be occur twice. Indexing starts at 0.

    For example mergebyidx2( [zip(["a","b"], [1,3]), zip(["c","d"] , [0,2])] )

    TODO: it doesn't work yet, but I'd like to fix it, just because I think its cool
    TODO: change it to be a generator! should be super simple! just put yield instead of append!
    """
    def next_(iter_):
        return next(iter_, (None, None))  # helper function

    zips = list(zips)  # necessary since we iterate over zips several times
    for (idx, lst) in zips:
        assert len(idx) == len(lst)
    merged = []  # we collect the merged list in here
    currents = list(map(next_, zips))  # a list of the heads of each of the input sequences
    result_len = reduce(lambda sum_, zip_: sum_ + len(zip_[0]), zips, 0)
    for idxres in range(result_len):
        for zipidx, (idx, val) in enumerate(currents):
            if idx == idxres:  # for each index we find the currently matching 'head'
                merged.append(val)  # if found, the corresponding value is appended to the merged list
                currents[zipidx] = next_(zips[idx])  # and the head is advanced
                break
    return merged


def mergebyidx(list1, list2, idx1, idx2):
    """Merges list l1 and list l2 into one list in increasing index order, where the indices are given by idx1
    and idx2, respectively. idx1 and idx2 are expected to be sorted. No index may be occur twice. Indexing starts at 0.

    For example mergebyidx( [a,b], [c,d], [1,3], [0,2] ) gives [c,a,d,b] )
    """
    assert (len(list1) == len(idx1) and len(list2) == len(idx2))
    result = []
    zip1 = zip(idx1, list1)
    zip2 = zip(idx2, list2)
    cur1 = next(zip1, (None, None))
    cur2 = next(zip2, (None, None))
    for idxres in range(len(list1) + len(list2)):
        if cur1[0] == idxres:
            result.append(cur1[1])
            cur1 = next(zip1, (None, None))
        elif cur2[0] == idxres:
            result.append(cur2[1])
            cur2 = next(zip2, (None, None))
        else:
            raise ValueError("missing index " + str(idxres) + " in given index ranges")
    return result


def _tuple2str(tuple_):
    """Returns a string that summarizes the given splittuple or aggregation tuple"""
    prefix = (str(tuple_.yields) + '@') if hasattr(tuple_, 'yields') and not tuple_.method == 'density' else ""
    return prefix + str(tuple_[1]) + '(' + str(tuple_[0]) + ')'


class Model:
    """An abstract base model that provides an interface to derive submodels
    from it or query density and other aggregations of it. It also defines stubs for those methods that actual models
    are required to implement.
    """

    # TODO: useful helper functions for dealing with fields and indexes:

    # todo: put it in models.py for reuse in all models?
    # precondition: it works for all data types...
    # @staticmethod
    # def _get_header(df):
    #     """ Returns suitable fields for this model from a given pandas dataframe.
    #     """
    #     fields = []
    #     for colname in df:
    #         column = df[colname]
    #         # if categorical of some sort, create discrete field from it
    #         if column.dtype == "category" or column.dtype == "object":
    #             domain = dm.DiscreteDomain()
    #             extent = dm.DiscreteDomain(sorted(column.unique()))
    #             field = md.Field(colname, domain, extent, 'string')
    #         # else it's numeric
    #         else:
    #             field = md.Field(colname, dm.NumericDomain(), dm.NumericDomain(column.min(), column.max()), 'numerical')
    #         fields.append(field)
    #     return fields

    def __str__(self):
        return (self.__class__.__name__ + " " + self.name + "':\n" +
                "dimension: " + str(self._n) + "\n" +
                "names: " + str([self.names]) + "\n")
                # "names: " + str([self.names]) + "\n" +
                # "fields: " + str([str(field) for field in self.fields]))

    def asindex(self, names):
        """Given a single name or a list of names of random variables, returns
        the indexes of these in the .field attribute of the model.
        """
        if isinstance(names, str):
            return self._name2idx[names]
        else:
            return [self._name2idx[name] for name in names]

    def byname(self, names):
        """Given a list of names of random variables, returns the corresponding
        fields of this model.
        """
        if isinstance(names, str):
            return self.fields[self._name2idx[names]]
        else:
            return [self.fields[self._name2idx[name]] for name in names]

    def isfieldname(self, names):
        """Returns true iff the single string or list of strings given as variables names are (all) names of random
        variables of this model.
        """
        if isinstance(names, str):
            names = [names]
        return all([name in self._name2idx for name in names])

    def inverse_names(self, names):
        """Given a sequence of names of random variables (or a single name), returns a sorted list of all names
        of random variables in this model which are _not_ in names. The order of names is the same as the order of
        fields in the model."""
        if isinstance(names, str):
            names = [names]
        names = set(names)
        return [name for name in self._name2idx if name not in names]

    def sorted_names(self, names):
        """Given a set, sequence or list of random variables of this model, returns a list of the
        same names but in the same order as the random variables in the model. If names contains
        """
        return utils.sort_filter_list(names, self.names)

    def __init__(self, name):
        self.name = name
        self.fields = []
        self.names = []
        self.data = pd.DataFrame([])
        self._aggrMethods = None
        self._n = 0
        self._name2idx = {}

    def _setempty(self):
        self.fields = []
        self._update()
        return self

    def _isempty(self):
        return self._n == 0

    def json_fields(self):
        return list(map(field_tojson, self.fields))

    def fit(self, df):
        """Fits the model to passed DataFrame

        This method must be implemented by any actual model that derives from the abstract Model class.

        Note that on return on this method the attribute .data must be filled with the appropriate data that
        was used to fit the model.

        Args:
            df: A pandas data frame that holds the data to fit the model to.

        Returns:
            The fitted model.
        """
        raise NotImplementedError("Implement this method in your model!")

    def marginalize(self, keep=None, remove=None):
        """Marginalizes random variables out of the model. Either specify which
        random variables to keep or specify which to remove.

        Note that marginalization depends on the domain of a random variable. That is: if
        its domain is bounded it is conditioned on this value (and marginalized out).
        Otherwise it is 'normally' marginalized out (assuming that the full domain is available).

        Arguments:
            keep: A list of names of random variables of this model to keep. All other random variables
                are marginalized out.
            remove: A list of names of random variables  of this model to marginalize out.

        Returns:
            The modified model.
        """
        logger.debug('marginalizing: ' + ('keep = ' + str(keep) if remove is None else ', remove = ' + str(remove)))

        if keep is not None and remove is not None:
            raise ValueError("You may only specify either 'keep' or 'remove', but non both.")
        if keep is not None:
            if keep == '*':
                keep = self.names
            if not self.isfieldname(keep):
                raise ValueError("invalid random variables names in argument 'keep': " + str(keep))
        elif remove is not None:
            if not self.isfieldname(remove):
                raise ValueError("invalid random variable names in argument 'remove': " + str(remove))
            keep = set(self.names) - set(remove)
        else:
            raise ValueError("Missing required argument: You must specify either 'keep' or 'remove'.")

        if len(keep) == self._n or self._isempty():
            return self
        if len(keep) == 0:
            return self._setempty()

        keep = self.sorted_names(keep)
        remove = self.inverse_names(keep)
        cond_out = [name for name in remove if self.byname(name)['domain'].isbounded()]

        # Data marginalization
        self.data = self.data.loc[:, keep]

        # Model marginalization
        # there are three cases of marginalization:
        # (1) unrestricted domain, (2) restricted, but not singular domain, (3) singular domain
        # we handle case (2) and (3) in ._conditionout, then case (1) in ._marginalizeout
        if len(cond_out) != 0:
            self._conditionout(cond_out)

        if len(keep) == self._n or self._isempty():
            return self

        return self._marginalizeout(keep)

    def _marginalizeout(self, keep):
        """Marginalizes the model such that only random variables with names in keep remain.

        This method must be implemented by any actual model that derived from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
          * keep is anything but a list of names of random variables of this model
          * keep is empty
          * the model itself is empty
          * keep contains all names of the model

        Moreover keep is guaranteed to be in the same order than the random variables of the model.
        """
        raise NotImplementedError("Implement this method in your model!")

    def condition(self, conditions):
        """Conditions this model according to the list of three-tuples
        (<name-of-random-variable>, <operator>, <value(s)>). In particular
        objects of type ConditionTuples are accepted and see there for allowed values.

        Note: This only restricts the domains of the random variables. To
        remove the conditioned random variable you need to call marginalize
        with the appropriate parameters.

        Returns:
            The modified model.
        """
        df = self.data
        for (name, operator, values) in conditions:
            operator = operator.lower()
            domain = self.byname(name)['domain']
            column = df[name]
            field = self.byname(name)

            if operator == 'in':
                domain.intersect(values)
                if field['dtype'] == 'numerical':
                     df = df.loc[column.between(*values, inclusive=True)]
                elif field['dtype'] == 'string':
                    df = df.loc[column.isin(values)]
                else:
                    raise TypeError("unsupported field type: " + str(field.dtype))
            else:
                # values is necessarily a single scalar value, not a list
                if operator == 'equals' or operator == '==':
                    domain.intersect(values)
                    df = df.loc[column == values]
                elif operator == 'greater' or operator == '>':
                    domain.setlowerbound(values)
                    df = df.loc[column > values]
                elif operator == 'less' or operator == '<':
                    domain.setupperbound(values)
                    df = df.loc[column < values]
                else:
                    raise ValueError('invalid operator for condition: ' + str(operator))
        self.data = df
        return self

    def _conditionout(self, remove):
        """Conditions the random variables with name in remove on their available, //not unbounded// domain and
        marginalizes them out.

        This method must be implemented by any actual model that derived from the abstract Model class.

        This method is guaranteed to be _not_ called if any of the following conditions apply:
          * remove is anything but a list of names of random-variables of this model
          * remove is empty
          * the model itself is empty
          * remove contains all names of the model

        Moreover when called remove is guaranteed to be in the same order than the random variables of the model.

        Note that we don't know yet how to condition on a non-singular domain (i.e. condition on interval or sets).
        As a work around we therefore:
          * for continuous domains: condition on (high-low)/2
          * for discrete domains: condition on the first element in the domain
        """
        raise NotImplementedError("Implement this method in your model!")

    def aggregate(self, method, mode="model"):
        """Aggregates this model using the given method and returns the
        aggregation as a list. The order of elements in the list matches the
        order of random variables in fields attribute of the model.

        Returns:
            The aggregation of the model. It always returns a list, even if it contains only a single value.
        """
        if self._isempty():
            raise ValueError('Cannot query aggregation of 0-dimensional model')

        # the data part can be aggregated without any model specific code and merged into the model results at the end
        # see my notes for how to calculate a single aggregation
        if mode == "data" or mode == "both":
            if method == 'maximum':
                # find observation with highest number of occurrences
                allcols = list(self.data.columns)
                grps = self.data.groupby(allcols)
                data_res = grps.size().argmax()

            elif method == 'average':
                # compute average of observations
                # todo: what if mean cannot be computed, e.g. categorical columns?
                data_res = self.data.mean(axis=0)

            else:
                raise ValueError("invalid value for method: " + str(method))

            # in case of 1-dimensional selection pandas returns a scalar, not a single-element list
            # we want a list in all cases, however
            if self._n == 1:
                data_res = [data_res]

            if mode == "data":
                return data_res

        # need index to merge results later
        other_idx = []
        singular_idx = []
        singular_names = []
        singular_res = []

        # 1. find singular fields (fields with singular domain)
        # any aggregation on such fields will yield the singular value of the domain
        for (idx, field) in enumerate(self.fields):
            domain = field['domain']
            if domain.issingular():
                singular_idx.append(idx)
                singular_names.append(field['name'])
                singular_res.append(domain.value())
            else:
                other_idx.append(idx)

        # quit early if possible
        if len(other_idx) == 0:
            model_res = singular_res
        else:
            # 2. marginalize singular fields out
            submodel = self.copy().marginalize(remove=singular_names)

            # 3. calculate 'unrestricted' aggregation on the remaining model
            try:
                other_res = submodel._aggrMethods[method]()
            except KeyError:
                raise ValueError("Your model does not provide the requested aggregation: '" + method + "'")

            # 4. clamp to values within domain
            for (idx, field) in enumerate(submodel.fields):
                other_res[idx] = field['domain'].clamp(other_res[idx])

            # 5. merge with singular results
            model_res = mergebyidx(singular_res, other_res, singular_idx, other_idx)

        if mode == "model":
            return model_res
        elif mode == "both":
            return (model_res, data_res)
        else:
            raise ValueError("invalid value for mode : ", str(mode))

    def density(self, names, values=None, mode="model"):
        """Returns the density at given point. You may either pass both, names
        and values, or only one list with values. In the latter case values is
        assumed to be in the same order as the random variables of the model.

        Args:
            values: may be anything that numpy.array accepts to construct from.
        """
        if self._isempty():
            raise ValueError('Cannot query density of 0-dimensional model')

        if values is not None and len(names) != len(values):
            raise ValueError('Length of names and values does not match.')

        if len(names) != self._n:
            raise ValueError('Not enough names/values provided. Require ' + str(self._n) +
                             ' but got only ' + str(len(names)) + '.')

        if values is None:
            # in that case the only argument holds the (correctly sorted) values
            values = names
        else:
            # names and values need sorting
            sorted_ = sorted(zip(self.asindex(names), values), key=lambda pair: pair[0])
            values = [pair[1] for pair in sorted_]

        # data frequency
        def filter(df, conditions):
            # apply all '==' filters in the sequence of conditions
            for (col_name, value) in conditions:
                df = df.loc[df[col_name] == value]
            return df

        if mode == "both" or mode == "data":
            cnt = len(filter(self.data, zip(self.names, values)))
            if mode == "data":
                return cnt

        p = self._density(values)
        if mode == "model":
            return p
        elif mode == "both":
            return (p, cnt)
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
        """Returns n samples drawn from the model."""
        if self._isempty():
            raise ValueError('Cannot sample from 0-dimensional model')

        samples = (self._sample() for i in range(n))
        return pd.DataFrame.from_records(samples, self.names)

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

        Note: you may use Model._defaultcopy() to get a copy of the 'abstract part' of an model
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
        mycopy._update()
        return mycopy

    def _condition_values(self, remove, pairflag=False):
        """Returns the list of values to condition on given a sequence of random variable names to condition on.

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
                cond_values.append(dvalue if domain.issingular() else (dvalue[1] - dvalue[0]) / 2)
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

    def _update(self):
        """Updates the _n, name2idx and names based on the fields in .fields"""
        # TODO: call it from aggregate, ... make it transparent to subclasses!? is that possible?
        self._n = len(self.fields)
        self._name2idx = dict(zip([f['name'] for f in self.fields], range(self._n)))
        self.names = [f['name'] for f in self.fields]

    def model(self, model, where=[], as_=None):
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

    def predict(self, predict, where=[], splitby=[], returnbasemodel=False, mode="both"):
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
        """
        if self._isempty():
            return (pd.DataFrame(),pd.DataFrame()) if mode == 'both' else pd.DataFrame()

        idgen = utils.linear_id_generator()

        # (1) derive base model, i.e. a model on all requested dimensions and measures, respecting filters
        predict_ids = []  # unique ids of columns in data frame. In correct order. For reordering of columns.
        predict_names = []  # names of columns as to be returned. In correct order. For renaming of columns.

        split_names = [f.name for f in splitby]  # name of fields to split by. Same order as in split-by clause.
        split_ids = [f.name + next(idgen) for f in
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
                id_ = _tuple2str(t) + next(idgen)
                aggrs.append(t)
                aggr_ids.append(id_)
                predict_names.append(_tuple2str(t))  # generate column name to return
                predict_ids.append(id_)
                basenames.update(t.name)

        basemodel = self.copy().model(basenames, where, '__' + self.name + '_base')

        # (2) derive a sub-model for each requested aggregation
        splitnames_unique = set(split_names)

        # for density: keep only those fields as requested in the tuple
        # for 'normal' aggregations: remove all random variables of other measures which are not also
        # a used for splitting, or equivalently: keep all random variables of dimensions, plus the one
        # for the current aggregation

        def _derive_aggregation_model(aggr):
            aggr_model = basemodel.copy()
            if aggr.method == 'density':
                return aggr_model.model(model=aggr.name)
            else:
                return aggr_model.model(model=list(splitnames_unique | set(aggr.name)))

        aggr_models = [_derive_aggregation_model(aggr) for aggr in aggrs]

        # (3) generate input for model aggregations,
        # i.e. a cross join of splits of all dimensions
        if len(splitby) == 0:
            input_frame = pd.DataFrame()
        else:
            def _get_group_frame(split, column_id):
                field = basemodel.byname(split.name)
                domain = field['domain'].bounded(field['extent'])
                try:
                    splitfct = sp.splitter[split.method.lower()]
                except KeyError:
                    raise ValueError("split method '" + split.method + "' is not supported")
                frame = pd.DataFrame({column_id: splitfct(domain.value(), split.args)})
                frame['__crossIdx__'] = 0  # need that index to cross join later
                return frame

            def _crossjoin(df1, df2):
                return pd.merge(df1, df2, on='__crossIdx__', copy=False)

            group_frames = map(_get_group_frame, splitby, split_ids)
            input_frame = reduce(_crossjoin, group_frames, next(group_frames)).drop('__crossIdx__', axis=1)

        # (4) query models and fill result data frame
        """ question is: how to efficiently query the model? how can I vectorize it?
            I believe that depends on the query. A typical query is consists of
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
        #result_list = [input_frame]
        model_result_list = [pd.DataFrame()]
        data_result_list = [pd.DataFrame()]
        for idx, aggr in enumerate(aggrs):
            aggr_results = []
            aggr_model = aggr_models[idx]
            if aggr.method == 'density':
                # TODO (1): this is inefficient because it recalculates the same value many times, when we split on more
                #  than what the density is calculated on
                # TODO: to solve it: calculate density only on the required groups and then join into the result table.
                # TODO (2): .density also returns the frequency of the specified observation. Calculating it like this
                #  is also super inefficient, since we really just need to group by all of the split-field and count
                #  occurrences - instead of iteratively counting occurrences, which involves a linear scan on the data
                #  frame for each call to density
                try:
                    # select relevant columns and iterate over it
                    ids = [split_name2id[name] for name in aggr.name]
                except KeyError as err:
                    raise ValueError("missing split-clause for field '" + str(err) + "'.")
                subframe = input_frame[ids]
                for _, row in subframe.iterrows():
                    res = aggr_model.density(aggr.name, row, mode=mode)
                    aggr_results.append(res)
            else:
                if len(splitby) == 0:
                    # there is no fields to split by, hence only a single value will be aggregated
                    # i.e. marginalize all other fields out
                    # TODO: can there ever be more fields left than the one(s) that is aggregated over?
                    if len(aggr.name) != aggr_model._n:  # debugging
                        raise ValueError("uh, interesting - check this out, and read the todo above this code line")
                    singlemodel = aggr_model.copy().marginalize(keep=aggr.name)
                    res = singlemodel.aggregate(aggr.method, mode=mode)
                    # reduce to requested dimension
                    res = res[singlemodel.asindex(aggr.yields)]
                    aggr_results.append(res)
                else:
                    for _, row in input_frame.iterrows():
                        pairs = zip(split_names, ['=='] * len(row), row)  # TODO: reuse ['=='] * len(row) for speedup
                        # derive model for these specific conditions
                        rowmodel = aggr_model.copy().condition(pairs).marginalize(keep=aggr.name)
                        res = rowmodel.aggregate(aggr.method, mode=mode)
                        # reduce to requested dimension
                        i = rowmodel.asindex(aggr.yields)
                        if mode == "both":
                            res = (res[0][i], res[1][i])
                        else:  # if mode == "model" or mode == "data":
                            res = res[i]
                        aggr_results.append(res)

            # generate DataSeries from it
            columns = [aggr_ids[idx]]
            if mode == "both":
                df = pd.DataFrame.from_records(aggr_results, columns=columns*2)
                model_result_list.append(df.iloc[:, 0])
                data_result_list.append(df.iloc[:, 1])
            else:
                df = pd.DataFrame(aggr_results, columns=columns)
                if mode == "data":
                    data_result_list.append(df)
                elif mode == "model":
                    model_result_list.append(df)
            #result_list.append(df)

        # (5) filter on aggregations?
        # TODO? actually there should be some easy way to do it, since now it really is SQL filtering

        # (6) collect all results into data frames
        if mode == "model" or mode == "both":
            model_result_list.append(input_frame)
            model_frame = pd.concat(model_result_list, axis=1)
            # (7) get correctly ordered frame that only contain requested fields
            model_frame = model_frame[predict_ids]  # flattens
            # (8) rename columns to be readable (but not unique anymore)
            model_frame.columns = predict_names
        # same as above
        if mode == "data" or mode == "both":
            data_result_list.append(input_frame)
            data_frame = pd.concat(data_result_list, axis=1)
            data_frame = data_frame[predict_ids]
            data_frame.columns = predict_names

        # DEBUG / DEVELOP
        #return_frame = pd.concat([input_frame, model_frame], axis=1)
        # (7) get correctly ordered frame that only contain requested fields
        #return_frame = return_frame[predict_ids]  # flattens
        # (8) rename columns to be readable (but not unique anymore)
        #return_frame.columns = predict_names
        # (9) return data frame or tuple including the basemodel
        #return (return_frame, basemodel) if returnbasemodel else return_frame

        # this is a little complicate, but I think necessary for a convenient interface
        if returnbasemodel:
            if mode == 'both':
                return (model_frame, data_frame, basemodel)
            elif mode == 'model':
                return (model_frame, basemodel)
            elif mode == 'data':
                return (data_frame, basemodel)
        else:
            if mode == 'both':
                return (model_frame, data_frame)
            elif mode == 'model':
                return model_frame
            elif mode == 'data':
                return data_frame
        raise ValueError("invalid mode: " + str(mode))
