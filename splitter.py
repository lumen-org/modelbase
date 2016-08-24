"""
This is the splitter module, which provides splitters and domains:
Splitters are functions that receive a domain and a number of parameters and 
return a split of the domain into values or subdomains.

Note that not all splitters work with all domains. See the documentation of
each for more.

@author: Philipp Lucas
"""
import numpy as np


class NumericDomain:
    def __init__(self, low, high):
        if low > high:
            raise ValueError()
        self.l = low
        self.h = high


def equidist(domain, args):
    """ Given a continuous numeric domain returns a list of n evenly spaced samples over
    the entire domain.

    Note that if the domain only consists of a single value, equiSample will
    also returns a single element list, regardless of n.
    """
    try:
        return [domain[0]] if domain[0] == domain[1] else np.linspace(domain[0], domain[1], args[0])
    except IndexError:
        return [domain[0]]


""" A map from 'method id' to the actual splitter function. """
splitter = {
    "equidist": equidist
}
