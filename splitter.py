"""
This is the splitter module, which provides splitters and domains:
Splitters are functions that receive a domain and a number of parameters and 
return a split of the domain into subdomains.

Note that not all splitters work with all domains. See the documentation of
each for more.

@author: Philipp Lucas
"""
import numpy as np

def equidist(domain, args):
    """ Given a continuous numeric domain returns a list of n evenly spaced samples over
    the entire domain.

    This function hence returns a list of domains. As each domain is a list itself it returns a list of lists.

    Note that if the domain only consists of a single value, this will returns a single element list, regardless of n.
    """
    try:
        return [domain[0]] if domain[0] == domain[1] else np.linspace(domain[0], domain[1], args[0])
    except TypeError:
        return [domain]
    except IndexError:
        return [domain[0]]

def equiinterval (domain, args):
    """ Splits the given continuous numeric into a list of n evently sized subdomains. The union of these subdomains
    is the original domain. """
    raise NotImplementedError()


def identity(domain, args):
    """ Given any domain returns the full domain itself. Note that for consistency it return a (single-element) list
    of domains. """
    return [domain]


""" A map from 'method id' to the actual splitter function. """
splitter = {
    "equidist": equidist,
    "identity": identity
}
