"""
Various utility functions

@author: phlpp
"""

def unique_list(iter_):
    """ Creates and returns a list from given iterable which only contains 
    each item once. Order is preserved. """
    ex = set()
    list_ = list()
    for i in iter_:
        if i not in ex:
            ex.add(i)
            list_.append(i)
    return list_