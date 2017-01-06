# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
Various utility functions

@author: Philipp Lucas
"""

import string
import random
from functools import wraps


def unique_list(iter_):
    """ Creates and returns a list from given iterable which only contains 
    each item once. Order is preserved. 
    """
    ex = set()
    list_ = list()
    for i in iter_:
        if i not in ex:
            ex.add(i)
            list_.append(i)
    return list_


def sort_filter_list(seq, reference):
    """Returns the list unique elements of seq that are contained in iterable reference in
    the same order as they appear in reference
    """
    seq = set(seq)
    return [val for val in reference if val in seq]


def random_id_generator(length=15, chars=string.ascii_letters + string.digits, prefix='__'):
    """ Generator for prefixed, random ids of given characters and given length."""
    while True:
        yield prefix + ''.join(random.choice(chars) for _ in range(length))


def linear_id_generator(prefix='_id', postfix=''):
    """ Generator for unique ids that are optionally pre- and postfixed."""
    num = 0
    while True:
        yield prefix + str(num) + postfix
        num += 1


def issorted(seq):
    """ Returns True iff seq is a strictly monotone increasing sequence."""
    return all(seq[i] < seq[i+1] for i in range(len(seq)-1))


def invert_indexes(idx, len_):
    """utility function that returns an inverted index list given a sorted 
    sequence of indexes, e.g. given [0,1,4] and len=6 it returns [2,3,5].
    """
    it = iter(idx)
    cur = next(it, None)
    inv = []
    for i in range(len_):        
        if i == cur:
            cur = next(it, None)
        else:
            inv.append(i)                
    return inv


def log_it(before, after):
    """decorator for convenient logging of whats happening. Pass a message to be printed
    before and after the functions is called."""
    def real_decorator(fct):
        @wraps(fct)
        def wrapper():
            end = "\n" if after is None else ""
            print(before, end=end)
            fct()
            print(after)
        return wrapper
    return real_decorator
