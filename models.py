"""
@author: Philipp Lucas

This module defines:

   * Model: an abstract base class for models.
   * Field: a class that represent random variables in a model.

It also defines models that implement that base model:

   * MultiVariateGaussianModel
"""
import pandas as pd
import math as math
import numpy as np
from numpy import pi, exp, matrix, ix_, nan
import copy as cp
from collections import namedtuple
from functools import reduce
from sklearn import mixture
import logging
import splitter as sp
import utils as utils
import domains as dm

# TODO: I don't know how to calculate aggregations beyond maximum, average and density of unrestricted multivariate gaussians
# e.g. what if we restrict the domain to >1? what is the average of such a distribution? how do we marginalize out such restricted fields?

# for fuzzy comparision.
# TODO: make it nicer?
eps = 0.000001

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

""" Development Notes (Philipp)

## how to get from data to model ##

1. provide some data file
2. open that data file
3. read that file into a tabular structure and guess its header, i.e. its
    columns and its data types
    * pandas dataframes can aparently guess headers/data types
4. use it to train a model
5. return the model

https://github.com/rasbt/pattern_classification/blob/master/resources/python_data_libraries.md !!!

### what about using a different syntax for the model, like the following:

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
            operator == 'in': A sequence of elements if the field is discrete. A two-element list [min, max] if the
                field is continuous.
            operator == 'equals' or operator == 'is': a single element of the domain of the field
            operator == 'greater': a single element that is set to be the new upper bound of the domain.
            operator == 'less': a single element that is set to be the new lower bound of the domain.
"""

"""class Domain:

    def issingular(self):
        raise "not implemented"

    def bounded(self, extent):
        raise "not implemented"

    def isbounded(self):
        raise "not implemented"

    def

class"""


def Field(name, domain, extent, dtype='numerical'):
    return {'name': name, 'domain': domain, 'extent': extent, 'dtype': dtype}


""" A constructor that returns 'Field'-dicts, i.e. a dict with three components
    as passed in:

    'name': the name of the field
TODO: fix/confirm notation
    'domain': the domain of the field, represented as a list as follows:
        if dtype == 'numerical':
            if domain is (partially) unbound: [-math.inf, math.inf], [val, math.inf] or [-math.inf, val], resp
            if domain in bound: A 2 element list of [min, max]
            if domain in singular: val (i.e. _not_ a list but a scalar)
        if dtype == 'string':
            if domain is unrestricted: math.inf (i.e. not a list, but the scalar value math.inf)
            if domain in restricted: [val2, val2, ...] i.e. a list of the possible values
            if domain in singular: [val]
            if domain is A list of the possible values.
    'dtype': the data type that the field represents. Possible values are: 'numerical' and 'string'
"""

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
    prefix = (str(tuple_.yields) + '@') if hasattr(tuple_, 'yields') else ""
    return prefix + str(tuple_[1]) + '(' + str(tuple_[0]) + ')'


def _issingulardomain(domain):
    """Returns True iff the given domain is singular, i.e. if it _is_ a single value."""
    return domain != math.inf and (isinstance(domain, str) or not isinstance(domain, (list, tuple)))  # \
    # or len(domain) == 1\
    # or domain[0] == domain[1]


def _isboundeddomain(domain):
    # its unbound if domain == math.inf or if domain[0] or domain[1] == math.inf
    if domain == math.inf:
        return False
    if _issingulardomain(domain):
        return True
    l = len(domain)
    if l > 1 and (domain[0] == math.inf or domain[1] == math.inf):
        return False
    return True

def _isunbounddomain(domain):
    return not _issingulardomain(domain) and not _isboundeddomain(domain)

def _boundeddomain(domain, extent):
    if domain == math.inf:
        return extent  # this is the only case a ordinal domain is unbound     
    if _issingulardomain(domain):
        return domain
    if len(domain) == 2 and (
            domain[0] == -math.inf or domain[1] == math.inf):  # hence this fulfills only for cont domains
        low = extent[0] if domain[0] == -math.inf else domain[0]
        high = extent[1] if domain[1] == math.inf else domain[1]
        return [low, high]
    return domain


def _clamp(domain, val, dtype):
    if dtype == 'string' and val not in domain:
        return domain[0]
    elif dtype == 'numerical':
        if val < domain[0]:
            return domain[0]
        elif val > domain[1]:
            return domain[1]
    return val


def _jsondomain(domain, dtype):
    """"Returns a domain that can safely be serialized to json, i.e. any infinity value is replaces by null."""
    if not _isunbounddomain(domain):
        return domain
    if dtype == 'numerical':
        l = domain[0]
        h = domain[1]
        return [None if l == -math.inf else l, None if h == math.inf else h]
    elif dtype == 'string':
        return None
    else:
        raise ValueError('invalid dtype of domain: ' + str(dtype))


class Model:
    """An abstract base model that provides an interface to derive submodels
    from it or query density and other aggregations of it.
    """

    @staticmethod
    def _get_header(df):
        """ Returns suitable fields for a model from a given pandas dataframe.
        """
        # TODO: at the moment this only works for continuous data.
        fields = []
        for column in df:
            field = Field(column, dm.NumericDomain(), dm.NumericDomain(df[column].min(), df[column].max()), 'numerical')
            fields.append(field)
        return fields

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
        """Returns true iff the name or names of variables given are names of
        random variables of this model.
        """
        if isinstance(names, str):
            names = [names]
        return all([name in self._name2idx for name in names])

    def __init__(self, name):
        self.name = name
        self.fields = []
        self.names = []
        self.data = []
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
        """Returns an adapted version of fields that in any case is JSON serializable. You cannot just use Model.fields
        because a fields domain may contain the value Infinity, which for our purpose cannot be correctly handled
        by json.dumps.
        """
        # copy everything but domain
        # special treat domain and return
        safe4json = []
        for field in self.fields:
            copy = cp.copy(field)
            copy['domain'] = field['domain'].tojson()
            copy['extent'] = field['extent'].tojson()
            safe4json.append(copy)
        return safe4json

    def fit(self, data):
        """Fits the model to the dataframe assigned to this model in at
        construction time.

        Returns:
            The modified model.
        """
        self.data = data
        self.fields = Model._get_header(self.data)
        self._fit()
        return self

    def _fit(self):
        raise NotImplementedError()

    def marginalize(self, keep=None, remove=None):
        """Marginalizes random variables out of the model. Either specify which
        random variables to keep or specify which to remove.

        Note that marginalization is depending on the domain of a random
        variable. That is: if nothing but a single value is left in the
        domain it is conditioned on this value (and marginalized out).
        Otherwise it is 'normally' marginalized out (assuming that the full
        domain is available)

        Returns:
            The modified model.
        """
        if self._isempty():
            return self
        logger.debug('marginalizing: ' + ('keep = ' + str(keep) if remove is None else ', remove = ' + str(remove)))

        if keep is not None:
            if keep == '*':
                keep = self.names
            if not self.isfieldname(keep):
                raise ValueError("invalid random variable names: " + str(keep))
        elif remove is not None:
            if not self.isfieldname(remove):
                raise ValueError("invalid random variable names: " + str(remove))
            keep = set(self.names) - set(remove)
        else:
            raise ValueError("cannot marginalize to zero-dimensional model")

        return self._marginalize(keep)

    def _marginalize(self, keep):
        raise NotImplementedError()

    def condition(self, conditions):
        """Conditions this model according to the list of three-tuples
        (<name-of-random-variable>, <operator>, <value(s)>). In particular
        ConditionTuples are accepted and see there for allowed values.

        Note: This only restricts the domains of the random variables. To
        remove the conditioned random variable you need to call marginalize
        with the appropriate paramters.

        Returns:
            The modified model.
        """
        if self._isempty():
            return self
        for (name, operator, values) in conditions:
            operator = operator.lower()
            domain = self.byname(name)['domain']
            if operator == 'equals' or operator == '==' or operator == 'in':
                domain.intersect(values)
            elif operator == 'greater' or operator == '>':
                domain.setlowerbound(values)
            elif operator == 'less' or operator == '<':
                domain.setupperbound(values)
            else:
                raise ValueError('invalid operator for condition: ' + str(operator))
        return self

    def aggregate(self, method):
        """Aggregates this model using the given method and returns the
        aggregation as a list. The order of elements in the list, matches the
        order of random variables in the models field.

        Returns:
            The aggregation of the model. It  always returns a list, even if the aggregation is one dimensional.
        """
        if self._isempty():
            return None

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
                #_issingulardomain(domain):
                singular_idx.append(idx)
                singular_names.append(field['name'])
                singular_res.append(domain.value())
                #singular_res.append(domain if field['dtype'] == 'numerical' else domain[0])
            else:
                other_idx.append(idx)

        # quit early if possible
        if len(other_idx) == 0:
            return singular_res

        # 2. marginalize singular fields out
        submodel = self.copy().marginalize(remove=singular_names)

        # 3. calculate 'unrestricted' aggregation on the remaining model
        try:
            other_res = submodel._aggrMethods[method]()
        except KeyError:
            raise ValueError("Your Model does not provide the requested aggregation: '" + method + "'")

        # 4. clamp to values within domain
        for (idx, field) in enumerate(submodel.fields):
            other_res[idx] = field['domain'].clamp(other_res[idx])
            #domain =
            #if _isboundeddomain(domain):
            #    other_res[idx] = _clamp(domain, other_res[idx], field['dtype'])

        # 5. merge with singular results
        return mergebyidx(singular_res, other_res, singular_idx, other_idx)

    def density(self, names, values=None):
        """Returns the density at given point. You may either pass both, names
        and values, or only one list with values. In the latter case values is
        assumed to be in the same order as the fields of the model.

        Args:
            values may be anything that numpy.matrix accepts to construct a vector from.
        """
        if self._isempty():
            return None

        if len(names) != self._n:
            raise ValueError(
                'Not enough names/values provided. Require ' + str(self._n) + ' got ' + str(len(names)) + '.')

        if values is None:
            # in that case the only argument holds the (correctly sorted) values
            values = names
        else:
            sorted_ = sorted(zip(self.asindex(names), values), key=lambda pair: pair[0])
            values = [pair[1] for pair in sorted_]
        return self._density(matrix(values).T)

    def sample(self, n=1):
        """Returns n samples drawn from the model."""
        if self._isempty():
            return None

        samples = (self._sample() for i in range(n))
        return pd.DataFrame.from_records(samples, self.names)

    def _sample(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def _update(self):
        """Updates the _n, name2idx and names based on the fields in .fields"""
        # TODO: call it from aggregate, ... make it transparent to subclasses!? is that possible?
        self._n = len(self.fields)
        self._name2idx = dict(zip([f['name'] for f in self.fields], range(self._n)))
        self.names = [f['name'] for f in self.fields]

    def model(self, model, where=[], as_=None):
        """Returns a model with name 'as_' that models the fields in 'model'
        respecting conditions in 'where'.

        Note that it does NOT create a copy, but modifies this model.

        Args:
            model:  A list of strings, representing the names of fields to 
                model. Its value may also be "*" or ["*"], meaning all fields
                of this model.
            where: A list of 'conditiontuple's, representing the conditions to
                model.
            as_: An optional string. The name for the model to derive. If set
                to None the name of the base model is used.

        Returns:
            The modified model.
        """
        self.name = self.name if as_ is None else as_
        return self.condition(where).marginalize(keep=model)

    def predict(self, predict, where=[], splitby=[], returnbasemodel=False):
        """ Calculates the prediction against the model and returns its result
        by means of a data frame.

        The data frame contains exactly those columns/random variables which
        are specified in 'predict'. Its order is preserved.

        Note that this does NOT modify the model it is called on.

        Args:
            predict: A list of names of fields (strings) and 'AggregationTuple's.
                This is hence the list of fields to be included in the returned
                dataframe.
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
            return pd.DataFrame()

        idgen = utils.linear_id_generator()

        # (1) derive base model, i.e. a model on all requested dimensions and measures, respecting filters
        predict_ids = []  # unique ids of columns in data frame. In correct order. For reordering of columns.
        predict_names = []  # names of columns as to be returned. In correct order. For renaming of columns.

        split_names = [f.name for f in splitby]  # name of fields to split by. Same order as in split-by clause.
        split_ids = [f.name + next(idgen) for f in
                     splitby]  # ids for columns for fields to split by. Same order as in splitby-clause.
        split_name2id = dict(zip(split_names, split_ids))  # maps split names to ids (for columns in data frames)

        aggrs = []  # list of aggregation tuples, in same order as in the predict-clause
        aggr_ids = []  # ids for columns fo fields to aggregate. Same order as in predict-clause

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
                # t is an aggregation tuple
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
                # _boundeddomain(field['domain'], field['extent'])
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
        result_list = [input_frame]
        for idx, aggr in enumerate(aggrs):
            aggr_results = []
            aggr_model = aggr_models[idx]
            if aggr.method == 'density':
                # TODO: this is inefficient because it recalculates the same value many times, when we split on more
                # than the density is calculated on
                try:
                    # select relevant columns and iterate over it
                    ids = [split_name2id[name] for name in aggr.name]
                except KeyError:
                    raise ValueError("missing split-clause for field '" + str(name) + "'.")
                subframe = input_frame[ids]
                for _, row in subframe.iterrows():
                    res = aggr_model.density(aggr.name, row)
                    aggr_results.append(res)
            else:
                if len(splitby) == 0:
                    # there is no fields to split by, hence only a single value will be aggregated
                    # i.e. marginalize all other fields out
                    singlemodel = aggr_model.copy().marginalize(keep=aggr.name)
                    res = singlemodel.aggregate(aggr.method)
                    # reduce to requested dimension
                    res = res[singlemodel.asindex(aggr.yields)]
                    aggr_results.append(res)
                else:
                    for _, row in input_frame.iterrows():
                        pairs = zip(split_names, ['=='] * len(row), row)
                        # derive model for these specific conditions
                        rowmodel = aggr_model.copy().condition(pairs).marginalize(keep=aggr.name)
                        res = rowmodel.aggregate(aggr.method)
                        # reduce to requested dimension
                        res = res[rowmodel.asindex(aggr.yields)]
                        aggr_results.append(res)

            df = pd.DataFrame(aggr_results, columns=[aggr_ids[idx]])
            result_list.append(df)

        # (5) filter on aggregations?
        # TODO? actually there should be some easy way to do it, since now it really is SQL filtering

        # (6) collect all results into one data frame
        return_frame = pd.concat(result_list, axis=1)

        # (7) get correctly ordered frame that only contain requested fields
        return_frame = return_frame[predict_ids]  # flattens

        # (8) rename columns to be readable (but not unique anymore)
        return_frame.columns = predict_names

        # (9) return data frame or tuple including the basemodel
        return (return_frame, basemodel) if returnbasemodel else return_frame


### ACTUAL MODEL IMPLEMENTATIONS

class MultiVariateGaussianModel(Model):
    """A multivariate gaussian model and methods to derive submodels from it
    or query density and other aggregations of it
    """

    def __init__(self, name):
        super().__init__(name)
        self._mu = nan
        self._S = nan
        self._detS = nan
        self._SInv = nan
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }

    def _fit(self):
        model = mixture.GMM(n_components=1, covariance_type='full')
        model.fit(self.data)
        self._model = model
        self._mu = matrix(model.means_).T
        self._S = matrix(model.covars_)
        self.update()

    def __str__(self):
        return ("Multivariate Gaussian Model '" + self.name + "':\n" +
                "dimension: " + str(self._n) + "\n" +
                "names: " + str([self.names]) + "\n" +
                "fields: " + str([str(field) for field in self.fields]))

    def update(self):
        """updates dependent parameters / precalculated values of the model"""
        self._update()
        if self._n == 0:
            self._detS = nan
            self._SInv = nan
        else:
            self._detS = np.abs(np.linalg.det(self._S))
            self._SInv = self._S.I

        assert (self._mu.shape == (self._n, 1) and
                self._S.shape == (self._n, self._n))

        return self

    def _conditionout(self, remove):
        """Conditions the random variables with name in remove on their available, //not unbounded// domain and marginalizes
        them out.

        Note that we don't know yet how to condition on a non-singular domain (i.e. condition on interval or sets).
        As a work around we therefore:
          * for continuous domains: condition on (high-low)/2
          * for discrete domains: condition on the first element in the domain
        """
        if len(remove) == 0 or self._isempty():
            return self
        if len(remove) == self._n:
            return self._setempty()

        j = sorted(self.asindex(remove))
        i = utils.invert_indexes(j, self._n)

        # collect singular values to condition out on
        condvalues = []
        for idx in j:
            field = self.fields[idx]
            domain = field['domain']
            dvalue = domain.value()
            assert (not domain.isunbounded())
            if field['dtype'] == 'numerical':
                condvalues.append(dvalue if domain.issingular() else (dvalue[1] - dvalue[0]) / 2)
            elif field['dtype'] == 'string':
                condvalues.append(domain[0])
                # actually it is: append(domain[0] if singular else domain[0])
                # TODO: we don't know yet how to condition on a not singular, but not unrestricted domain.
            else:
                raise ValueError('invalid dtype of field: ' + str(field['dtype']))

        # store old sigma and mu
        S = self._S
        mu = self._mu
        # update sigma and mu according to GM script
        self._S = MultiVariateGaussianModel._schurcompl_upper(S, i)
        self._mu = mu[i] + S[ix_(i, j)] * S[ix_(j, j)].I * (condvalues - mu[j])
        self.fields = [self.fields[idx] for idx in i]
        return self.update()

    def _marginalizeout(self, keep):
        if len(keep) == self._n or self._isempty():
            return self
        if len(keep) == 0:
            return self._setempty()

        # i.e.: just select the part of mu and sigma that remains
        keepidx = sorted(self.asindex(keep))
        self._mu = self._mu[keepidx]
        self._S = self._S[np.ix_(keepidx, keepidx)]
        self.fields = [self.fields[idx] for idx in keepidx]
        return self.update()

    def _marginalize(self, keep):
        if len(keep) == self._n:
            return self
        # there are three cases of marginalization:
        # (1) unrestricted domain, (2) restricted, but not singular domain, (3) singular domain
        # we handle case (2) and (3) in ._conditionout, then case (1) in ._marginalizeout
        condout = [field['name'] for idx, field in enumerate(self.fields)
                   if (field['name'] not in keep) and field['domain'].isbounded()]
        return self._conditionout(condout)._marginalizeout(keep)

    def _density(self, x):
        """Returns the density of the model at point x.

        Args:
            x: a Scalar or a _column_ vector as a numpy matrix.
        """
        xmu = x - self._mu
        return ((2 * pi) ** (-self._n / 2) * (self._detS ** -.5) * exp(-.5 * xmu.T * self._SInv * xmu)).item()

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        # _mu is a np matrix, but we want to return a list
        return self._mu.T.tolist()[0]

    def _sample(self):
        # TODO: let it return a dataframe?
        return self._S * np.matrix(np.random.randn(self._n)).T + self._mu

    def copy(self, name=None):
        # TODO: this should be as lightweight as possible!
        name = self.name if name is None else name
        mycopy = MultiVariateGaussianModel(name)
        mycopy.data = self.data
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy._mu = self._mu.copy()
        mycopy._S = self._S.copy()
        mycopy.update()
        return mycopy

    @staticmethod
    def custom_mvg(sigma, mu, name):
        """Returns a MultiVariateGaussian model that uses the provided sigma, mu and name.

        Note: The domain of each field is set to (-10,10).

        Args:
            sigma: a suitable numpy matrix
            mu: a suitable numpy row vector
        """

        if not isinstance(mu, matrix) or not isinstance(sigma, matrix) or mu.shape[1] != 1:
            raise ValueError("invalid arguments")
        model = MultiVariateGaussianModel(name)
        model._S = sigma
        model._mu = mu
        model.fields = [Field(name="dim" + str(idx),
                              domain=dm.NumericDomain(),
                              extent=dm.NumericDomain(mu[idx].item() - 2, mu[idx].item() + 2))
                        for idx in range(sigma.shape[0])]
        model.update()
        return model

    @staticmethod
    def normal_mvg(dim, name):
        sigma = matrix(np.eye(dim))
        mu = matrix(np.zeros(dim)).T
        return MultiVariateGaussianModel.custom_mvg(sigma, mu, name)

    @staticmethod
    def _schurcompl_upper(M, idx):
        """Returns the upper Schur complement of matrix M with the 'upper block'
        indexed by idx.
        """
        # derive index lists
        i = idx
        j = utils.invert_indexes(i, M.shape[0])
        # that's the definition of the upper Schur complement
        return M[ix_(i, i)] - M[ix_(i, j)] * M[ix_(j, j)].I * M[ix_(j, i)]


if __name__ == '__main__':
    import pdb

    # foo = MultiVariateGaussianModel.normalMVG(5,"foo")
    sigma = matrix([
        [1.0, 0.6, 0.0, 2.0],
        [0.6, 1.0, 0.4, 0.0],
        [0.0, 0.4, 1.0, 0.0],
        [2.0, 0.0, 0.0, 1.]])
    mu = np.matrix([1.0, 2.0, 0.0, 0.5]).T
    foo = MultiVariateGaussianModel.custom_mvg(sigma, mu, "foo")
    foocp = foo.copy("foocp")
    print("\n\nmodel 1\n" + str(foocp))
    foocp2 = foocp.model(['dim1', 'dim0'], as_="foocp2")
    print("\n\nmodel 2\n" + str(foocp2))

    res = foo.predict(predict=['dim0'], splitby=[SplitTuple('dim0', 'equiDist', [5])])
    print("\n\npredict 1\n" + str(res))

    res = foo.predict(predict=[AggregationTuple(['dim1'], 'maximum', 'dim1', []), 'dim0'],
                      splitby=[SplitTuple('dim0', 'equiDist', [10])])
    print("\n\npredict 2\n" + str(res))

    res = foo.predict(predict=[AggregationTuple(['dim0'], 'maximum', 'dim0', []), 'dim0'],
                      where=[ConditionTuple('dim0', 'equals', 1)], splitby=[SplitTuple('dim0', 'equiDist', [10])])
    print("\n\npredict 3\n" + str(res))

    res = foo.predict(predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
                      splitby=[SplitTuple('dim0', 'equiDist', [10])])
    print("\n\npredict 4\n" + str(res))

    res = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        where=[ConditionTuple('dim0', 'greater', -1)])
    print("\n\npredict 5\n" + str(res))

    res = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        where=[ConditionTuple('dim0', 'less', -1)])
    print("\n\npredict 6\n" + str(res))

    res = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        where=[ConditionTuple('dim0', 'less', 0), ConditionTuple('dim2', 'equals', -5.0)])
    print("\n\npredict 7\n" + str(res))

    res, base = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10]), SplitTuple('dim1', 'equiDist', [7])],
        where=[ConditionTuple('dim0', 'less', -1), ConditionTuple('dim2', 'equals', -5.0)],
        returnbasemodel=True)
    print("\n\npredict 8\n" + str(res))

    res, base = foo.predict(
        predict=[AggregationTuple(['dim0'], 'average', 'dim0', []), AggregationTuple(['dim0'], 'density', 'dim0', []),
                 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        # where=[ConditionTuple('dim0', 'less', -1), ConditionTuple('dim2', 'equals', -5.0)],
        returnbasemodel=True)
    print("\n\npredict 9\n" + str(res))

    res, base = foo.predict(
        predict=['dim0', 'dim1', AggregationTuple(['dim0', 'dim1'], 'average', 'dim0', []),
                 AggregationTuple(['dim0', 'dim1'], 'average', 'dim1', [])],
        splitby=[SplitTuple('dim0', 'identity', []), SplitTuple('dim1', 'equiDist', [4])],
        where=[ConditionTuple('dim0', '<', 2), ConditionTuple('dim0', '>', 1)],
        returnbasemodel=True)
    print("\n\npredict 10\n" + str(res))

    # print("\n\n" + str(base) + "\n")
