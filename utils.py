# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
Various utility functions

@author: Philipp Lucas
"""

import string
import random
from functools import wraps


def equiweightedintervals(seq, k, is_sorted=False, bins=False, eps=0.1):
    """Divide seq into k intervals where each interval contains 1/k-th of the data.

    Args:
        bins:
            if bins == False:  Returns the value borders of the intervals as a sequence of 2-tuples of the
                form [min, max].
            else: Returns a the sequence of interval borders.

        eps: the range of seq is extended by eps% on each side of the interval to include the min or max values of seq.

    Note: seq _will_ be altered if is_sorted is False.

    Note: the intervals cannot be guaranteed to each hold the same or even similar number of elements if there are
    many elements in seq that occur more than once.
    """

    intervals = []

    if k <= 0:
        raise ValueError("k must be > 0")

    n = len(seq)
    if n < k:
        raise ValueError("cannot generate more intervals than the sequence is long")

    # sort in place
    if not is_sorted:
        list.sort(seq)

    eps *= (seq[-1]-seq[0])*0.01
    aggr_borders = [seq[0]-eps]
    borders = [seq[0]-eps]
    l = n/k
    leps = l-eps  # is that nice?
    cnt = 0
    for val in seq:
        cnt += 1
        if cnt >= leps:  # add a new border
            cnt -= l
            borders.append(val)
            aggr_borders.append(val)
            if len(borders) % 2 == 0:  # add a new interval
                high = borders.pop()
                low = borders.pop()
                intervals.append((low, high))
                borders.append(val)

    return aggr_borders if bins else intervals


def shortest_interval(seq):
    """ Given a sequence of intervals (i.e. a 2-tuple or a 2-element list), return the index of the shortest interval"""
    width = [s[1] - s[0] for s in seq]
    try:
        min_ = min(width)
        idx = width.index(min_)
        return idx
#        return seq[idx]
    except ValueError:
        # min() arg is an empty sequence
        return None


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


if __name__ == '__main__':
    import numpy as np
    vec = list(np.floor(np.random.rand(18) * 100))  # vector of random numbers
    k = 6  # number of intervals
    res = equiweightedintervals(vec, k)#, bins=True)
    print(res)
    print(vec)
    #shortest = shor