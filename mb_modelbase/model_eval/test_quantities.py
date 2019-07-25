# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas
"""

import numpy as np

"""Contains various test quantities for posterior predictive checks."""


def quantile(a, q):
    return np.quantile(a, q, interpolation='nearest')


def mean(a):
    return np.mean(a)


def median(a):
    return quantile(a, 0.5)


def minimum(a):
    return np.min(a)


def maximum(a):
    return np.max(a)




