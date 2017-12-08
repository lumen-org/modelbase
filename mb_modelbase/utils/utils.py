# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
Various utility functions

@author: Philipp Lucas
"""

import string
import random
from functools import wraps
from numpy import matrix, ix_, isfinite, linalg
from xarray import DataArray


def assert_all_psd(S, len_num):
    for s in S.reshape(-1, len_num, len_num):
        assert(is_psd(s))


def is_psd(a):
    try:
        linalg.cholesky(a)
    except linalg.LinAlgError:
        return False
    else:
        return True


def numpy_to_xarray_params(p, mu, Sigma, cat_levels, cat, num):
    """Translates numpy based parameters into suitably labelled xarray DataArrays and returns them.

    Args:
        p, mu, Sigma: numpy arrays that contain the mean parameters of a cg-model.
        cat_levels: dict of sequences that contain the levels per categoricals dimension (indexed by name)
        cat: labels of categorical dimensions
        num: labels of numerical dimensions.

    Note: the order of labels, dimensions, ... must all correspond to each other.
    """

    # make sure shape is correct:
    # TODO: the shape of p sometimes doesn't match with cat_levels and cat ... BUG!?
    s = tuple(len(l) for l in cat_levels)
    p = p.reshape(s)
    # now make it to xarray object
    p = DataArray(data=p, coords=cat_levels, dims=cat)

    coords = cat_levels + [num]
    dims = cat + ['mean']
    mu.shape = [len(e) for e in coords]
    mu = DataArray(data=mu, coords=coords, dims=dims)

    coords = cat_levels + [num] + [num]
    dims = cat + ['S1', 'S2']
    Sigma.shape = [len(e) for e in coords]
    Sigma = DataArray(data=Sigma, coords=coords, dims=dims)
    return p, mu, Sigma


def validate_opts(opts, allowed):
    """Validates a dictionary of categorical options with respect to provided dictionary of allowed options and their values.
    Raises a ValueError if anything is wrong."""
    cpy = opts.copy()
    for opt_name, allowed_vals in allowed.items():
        if opt_name in cpy:
            val = cpy[opt_name]
            if val not in allowed_vals:
                raise ValueError(
                    'invalid value "{1!s}" for argument {0}. Allowed is {2!s}'.format(opt_name, val, allowed_vals))
            del cpy[opt_name]

    # there should nothing be left
    if len(cpy) > 0:
        raise ValueError('unrecognized argument names: {0!s}'.format(list(cpy.keys())))


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


def rolling_1d_mean(seq):
    return [(seq[i]+seq[i+1])/2 for i in range(len(seq)-1)]


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
    """Returns the list of unique elements of seq that are contained in iterable reference in
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

    This is a special case of invert_sequence.
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


def invert_sequence(seq, base):
    """utility function that returns an inverted sequence given a base sequence and sorted
    sequence of objects (with respect to base).
    """
    it = iter(seq)
    cur = next(it, None)
    inverted = []
    for val in base:
        if val == cur:
            cur = next(it, None)
        else:
            inverted.append(val)
    return inverted


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


def schur_complement(M, idx):
    """Returns the upper Schur complement of array_like M with the 'upper block'
    indexed by idx.
    """
    M = matrix(M, copy=False)  # matrix view on M
    # derive index lists
    i = idx
    j = invert_indexes(i, M.shape[0])
    # that's the definition of the upper Schur complement
    return M[ix_(i, i)] - M[ix_(i, j)] * M[ix_(j, j)].I * M[ix_(j, i)]


def is_running_in_debug_mode():
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


def truncate_string(str_, trim_length=500):
    if trim_length == 0:
        return str_
    else:
        return (str_[:trim_length] + ' ...') if len(str_) > trim_length else str_


def no_nan(nparr):
    return isfinite(nparr).any()


if __name__ == '__main__':
    pass