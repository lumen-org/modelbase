"""
Various utility functions

@author: Philipp Lucas
"""

import string
import random

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
    return all( seq[i] < seq[i+1] for i in range(len(seq)-1))
    
def invert_indexes (idx, len_) :
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