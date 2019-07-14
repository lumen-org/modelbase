# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

This module provides basic types required for describing and working with models and data, in particular:

    * Field: an attribute/dimension/random variable

    * Field Usages: An usage of a (number of) Fields to derive a (set of) values/intervals

        * Aggregation: Aggregation of one or more fields. In a sense it is a summary across the given fields.

        * Split: Splits divide the domain of the referenced (single) field and generate a set of values/intervals of
        the domain of the field. Typcially, this is done because input values of that field are required.

        * Density/Probability: Is a particular type of aggregation that represent queries for the density
        value/probability.

    See below for details!
"""
from collections import namedtuple


AggregationMethods = {'maximum', 'average', 'density', 'probability'}  # possible method of an aggregation
SplitMethods = {'elements', 'identity', 'equiinterval', 'equidist', 'data'}  # possible method of a split

AggregationTuple = namedtuple('AggregationTuple', ['name', 'method', 'yields', 'args'])
"""An aggregation tuple describes an aggregation.

See also `Aggregation`, `Density` and `Probability` for the recommended and more convenient way of creating different 
types of aggregation tuples. 

Note that is is optional to used these named tuples. Normal tuples may also be use to represent aggregations. 

Attributes:
    name : sequence of string 
        The names of the fields that is aggregated over.
    method : string, see `AggregationMethods`
        The method to use for aggregation.
    yields : string
        The name of the field that the resulting value will be of. 
    args 
        Additional arguments passed to the aggregation function.
"""

SplitTuple = namedtuple('SplitTuple', ['name', 'method', 'args'])
"""A split tuple details how a field of a model is split by.

See also `Split` for the recommended and more convenient way of creating split tuples.

Attributes:
    name : string
        The  name of the field to split.
    method : string, see `SplitMethods`
        The method to use to split.
    args
        Additional arguments passed to the split function.
"""
NAME_IDX = 0
METHOD_IDX = 1
YIELDS_IDX = 2
ARGS_IDX = 2

ConditionTuple = namedtuple('ConditionTuple', ['name', 'operator', 'value'])
"""A condition tuple describes the details of how a field of model is conditioned.

See also `Condition` for the recommended and more convenient way of creating condition tuples.

Attributes:
    name : string
        The name of field to condition.
    operator : ['in', 'equals', '==', 'greater', 'less']
        The operator of the condition.
    value : its allowed values depend on the value of operator        
        operator is 'in': A single element (to set the domain singular) or a sequence of elements (if the field is 
            discrete), or a two-element list [min, max] if the field is continuous. 
        operator is 'equals' or '==': a single element
        operator == 'greater': a single element that is set to be the new upper bound of the domain.
        operator == 'less': a single element that is set to be the new lower bound of the domain.
"""
OP_IDX = 1
VALUE_IDX = 2


def Field(name, domain, extent, independent, dtype='numerical'):
    """ A factory for 'Field'-dicts.

    Fields represent the dimensions, random variables or attributes of models.

    Attributes: a field has the following attributes:
        'name': Same as the argument to this function.
        'domain': Same as the argument to this function.
        'extent': Same as the argument to this function.
        'independent' : Same as the argument to this function.
        'dtype': Same as the argument to this function.
        'default_value': See `Model`.
        'default_subset': See `Model`.


    Args:
        name : string
            The name of the field.
        domain : `dm.Domain`
            The domain of the field.
        extent : `dm.Domain`
            The extent of the field. May not be unbounded. The extent of the field that will be used as a fallback
            for domain if domain is unbounded but a value for domain is required
        independent : [True, False]
            Describes if the according variable is an independent variable
        dtype : ['numerical', 'string'] , optional.
            A string identifier of the data type of this field.


    Returns : dict
        The constructed 'field dictionary'.
    """
    if not extent.isbounded():
        raise ValueError("extents must not be unbounded")
    if dtype not in ['numerical', 'string']:
        raise ValueError("dtype must be 'string' or 'numerical'")
    field = {'name': name, 'domain': domain, 'extent': extent, 'dtype': dtype, 'hidden': False, 'default_value': None,
             'default_subset': None, 'independent': independent}
    return field


def Aggregation(base, method='maximum', yields=None, args=None):
    """A factory for 'AggregationTuples'.

    This is the preferred way to construct `AggregationTuple`s as it provides some convenience over the raw
    method, such as filling in missing values, deriving suitable defaults and some value/type checking.

    See also `AggregationTuple`.

    Arguments:
        base : A single or a sequence of field names or fields
            The fields to aggregate over.
        method : string, see `AggregationMethods`, optional.
            The method to use for aggregation. Defaults to 'maximum'.
        yields : string, optional
            The name of the field that the resulting value will be of. Needs not to be provided for methods
            'probability' and 'density'.
        args : Any, optional.
            Additional arguments passed to the aggregation function. Defaults to None
    """
    name = to_name_sequence(base)
    if yields is None:
        yields = "" if method == 'density' or method == 'probability' else name[0]
    if args is None:
        args = []
    if method not in AggregationMethods:
        raise ValueError('invalid aggregation method: ' + str(method))
    return AggregationTuple(name, method, yields, args)


def Density(base):
    """A factory for density aggregations.

    This is the preferred way to construct an `AggregationTuple` that encodes a density query.

    See also `AggregationTuple`.

    base : a single or a sequence of `Field` or string
            The names to query density over.
    """
    return Aggregation(base, method='density')


def Probability(base):
    """A factory for probability aggregations.

    This is the preferred way to construct an `AggregationTuple` that encodes a probability query.

    See also `AggregationTuple`.

    Args:
        base : a single or a sequence of `Field` or string
            The names to query probability over.
    """
    return Aggregation(base, method='probability')


def Split(base, method=None, args=None):
    """A factory for splits.

    This is the preferred way to create `SplitTuple`s as it provides some convenience over the raw
    method, such as filling in missing values, deriving suitable defaults and some value/type checking.

    There is some auto-completion available, where meaningfully possible, as follows:

        default methods by data type (dtype)
            * 'elements' for 'string' dtype
            * 'equiinterval' for 'numerical' dtype

        default arguments by method:
            * [25] for 'equiinterval' and 'equidist'
            * [] for 'identity' and 'elements'

    Args:
        base : a single or a sequence of `Field` or string
            The names to query probability over.
        method : string, see `SplitMethods`, optional.
            The method to use to split.
        args : any, optional
            Additional arguments passed to the split function. Default

    """
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


def Condition(base, operator, value):
    """A factory for 'ConditionTuple's

    This is the preferred way to construct `ConditionTuple`s as it provides some convenience over the raw
    method, such as filling in missing values, deriving suitable defaults and some value/type checking.

    See also `ConditionTuple`.

    Args:
        base : Field or string
            The field to condition on.
        operator : string
            The operator of the condition.
        value : Any
            See `ConditionTuple`.
    """
    return ConditionTuple(_name_from_field(base), operator, value)


""" Utility functions to transform and normalize fields and field usages """


def _name_from_field(base):
    """Base may either be a field name or a `Field`. It returns the fields name in either cases."""
    if isinstance(base, str):
        return base
    try:
        return base['name']
    except KeyError:
        raise TypeError('Base must be a Field-dict or a name')


def _is_single_name_or_field(obj):
    """Return True iff `obj` is a string or a dictionary.

    This is used as a heuristic to determine whether or not `obj` is a name of a `Field` or a `Field`.

    Args:
        obj : Any
            The object to check.
    """
    return isinstance(obj, (str, dict))


def _to_sequence(obj):
    """Makes `obj` a one-element sequence, if _is_single_name_or_field(obj), else it returns the object itself."""
    return [obj] if _is_single_name_or_field(obj) else obj


def to_name_sequence(obj):
    """Transform `obj` into a sequence of field names and return it.

    Args:
        obj : sequence or single of `Field` or string
            Field(s) or name(s) of Fields to convert.

    Returns
        sequence of str
    """
    return list(map(_name_from_field, _to_sequence(obj)))


# def field_usage_type(fu):
#     """Returns the 'type' of the FieldUsage `fu`
#
#     Args:
#         fu : FieldUsage
#
#     Returns: str
#         Returns a string to identify the type of the field usage, as follows:
#         * 'split' if it is any split
#         * 'density' if it is an aggregation with method 'density' OR 'probability'
#         * 'aggregation' if it is an aggregation with method 'maximum' OR 'average'
#     """
#     return fu[METHOD_IDX]