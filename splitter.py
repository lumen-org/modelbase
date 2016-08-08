"""
This is the splitter module, which provides splitters: functions that receive 
a domain (see module domain) and a number of parameters and return a split of 
the domain into values or subdomains.

Note that not all splitters work with all domains. See the documentation of 
each for more.

@author: Philipp Lucas
"""
import numpy as np

# def equiSplit (domain, n)

class NumericDomain:
    def __init__ (self, low, high):
        if low > high:
            raise ValueError()
        self.l = low
        self.h = high

def equiSample (domain, n):
    """ Given a continuous numeric domain returns a list of n evenly spaced samples over
    the entire domain. 
    
    Note that if the domain only consists of a single value, equiSample will 
    also returns a single element list, regardless of n."""
    return domain.l if domain.l == domain.h else np.linspace(domain.l, domain.h, n)

""" A map from 'method id' to the actual splitter function """
splitter = {  
    "equiDist" : equiSample
}