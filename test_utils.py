# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for cond_gaussians.py
"""

import unittest
import numpy as np
import pandas as pd

import utils


class TestUtils(unittest.TestCase):

    def test_equiweightedintervals(self):
        for i in range(100):
            for k in range(1, 13):
                # k = 5  # number of intervals
                vec = list(np.floor(np.random.rand(i+k) * 100))  # vector of random numbers
                res = utils.equiweightedintervals(vec, k)
                self.assertEqual(len(res), k)
                res = utils.equiweightedintervals(vec, k, bins=True)
                self.assertEqual(len(res), k + 1)

    def test_shortest_interval(self):
        # TODO
        pass

    def test_invert_indexes(self):
        # already tested in test_models.py
        pass


if __name__ == '__main__':
    unittest.main()
