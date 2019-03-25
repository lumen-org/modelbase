# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
This is the splitter module, which provides splitters and domains:
Splitters are functions that receive a domain and a number of parameters and 
return a split of the domain into subdomains.

Note that not all splitters work with all domains. See the documentation of
each for more.


Scalar or interval-valued return values

    Splits return different _types_ of values:
     * scalars (i.e. single elements of the splitted domain), or
     * domains  (i.e. a tuple, such as (min, max) for quantitative domains, or (el1, el2) for categorical domains)

    The return type different between the split functions, and in the future there may be flags to configure what to
    return. Note, that generally and 'up-cast' is possible, i.e. returning a domain instead of a scalar.

TODO:
    need to clean up domains. documentation of functions and their implementation does not match. Also, we lack
    a good strategy to distinghuish scalar and domain-values splits and how to deal with them in predict.

@author: Philipp Lucas
"""
import numpy as np

# TODO: why is this not working on real domains, but on lists and such?
# what is more elegant? change it if necessary


def equidist(domain, args):
    """ Given a continuous numeric domain returns a list of n evenly spaced samples over
    the entire domain.

    Note: This function returns a list of domains. As each domain is a list itself it returns a list of lists.
    Note: if the domain only consists of a single value, this will return a single element list, regardless of n.
    """
    try:
        return [domain[0]]*(args[0] > 0) if domain[0] == domain[1] else np.linspace(domain[0], domain[1], args[0])
        #TODO: future if we use domains: return [domain[0]] if domain.issingular() else np.linspace(domain[0], domain[1], args[0])
    except TypeError:
        # domain is not a list
        return [domain]
    except IndexError:
        # domain only has one element
        return [domain[0]]


def equiinterval(domain, args):
    """ Splits the given continuous numeric into a list of n evently sized subdomains. The union of these subdomains
    is the original domain.
    Note: if the domain only consists of a single value, this will return a single element list, regardless of n."""
    n = args[0]
    if n > 0:
        points = equidist(domain, [n+1])
        if len(points) == 1:
            points.append(points[0])  # add another point to make sure we get any interval at all
        return [(points[i], points[i+1]) for i in range(len(points)-1)]
    elif n == 0:
        return []
    else:
        raise ValueError("Number of samples, " + str(n) + ", must be non-negative.")


def identity(domain, args):
    """ Given any domain returns the full domain itself. Note that for consistency it return a (single-element) list
    of domains. """
    return [domain]
    # OLD:
    # return [domain]


def elements(domain, args):
    """ Splits the given discrete domain into it's single elements and returns these. Thus, it returns a list
    of these elements."""
    if isinstance(domain, str):
        return [(domain,)]
    else:
        return [(e,) for e in domain]
    # OLD:
    #     if isinstance(domain, str):
    #         return [(domain,)]
    #         # raise TypeError('domain must be a list of values, not a single value')
    #     else:
    #         #return domain
    #         return [(e,) for e in domain]


""" A map from 'method id' to the actual splitter function. """
splitter = {
    "equidist": equidist,
    "equiinterval": equiinterval,
    "identity": identity,
    "elements": elements
}

"""A dict from `method_id` to a method's return type ('scalar' or 'domain')."""
return_types = {
    "equidist": 'scalar',
    "equiinterval": 'domain',
    "identity": 'domain',
    "elements": 'domain'
}

