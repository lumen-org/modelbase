# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for cond_gaussians.py
"""

import unittest
import numpy as np
from random import shuffle
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
        seq = [(0, 0), (0, 1), (-1, -0.5)]
        self.assertEqual(0, utils.shortest_interval(seq))

        seq = [(0, 1), (-1, -0.5)]
        self.assertEqual(1, utils.shortest_interval(seq))

        seq = [(0, 5), (0, .1), (-1, -0.5)]
        self.assertEqual(1, utils.shortest_interval(seq))

        seq = []
        with self.assertRaises(IndexError):
            utils.shortest_interval(seq)

        seq = [(-1, -1), (10, 20)]
        self.assertEqual(0, utils.shortest_interval(seq))

    def test_unique_list(self):
        self.assertEqual(list(range(10)), utils.unique_list(range(10)))

        seq = [1, 2, 3, 3, 4, 3, 1, 2, -10, 0, 0]
        self.assertEqual([1, 2, 3, 4, -10, 0], utils.unique_list(seq))

        self.assertEqual([], utils.unique_list([]))

        self.assertEqual([10.0], utils.unique_list([10.0]*20))

    def test_sort_filter_list(self):
        reference = list('ABCDEFGHIJK')
        other = list('UVWXYZ')
        n = len(reference)
        for i in range(25):
            lst = reference[0:n//2]
            shuffle(lst)
            self.assertEqual(reference[0:n//2], utils.sort_filter_list(lst, reference))

            lst = reference[n//4:-n//4]
            shuffle(lst)
            self.assertEqual(reference[n//4:-n//4], utils.sort_filter_list(lst, reference))

            lst = reference[n//2:]
            shuffle(lst)
            self.assertEqual(reference[n//2:], utils.sort_filter_list(lst, reference))

            lst = reference[0:n // 2] + other
            shuffle(lst)
            self.assertEqual(reference[0:n // 2], utils.sort_filter_list(lst, reference))

            lst = reference[n // 4:-n // 4] + other
            shuffle(lst)
            self.assertEqual(reference[n // 4:-n // 4], utils.sort_filter_list(lst, reference))

            lst = reference[n // 2:] + other
            shuffle(lst)
            self.assertEqual(reference[n // 2:], utils.sort_filter_list(lst, reference))

    def test_issorted(self):
        self.assertTrue(utils.issorted([]))
        self.assertTrue(utils.issorted([1]))
        self.assertTrue(utils.issorted([1, 2]))
        self.assertFalse(utils.issorted([2, 1]))
        self.assertTrue(utils.issorted(list(range(10))))
        self.assertFalse(utils.issorted(list(range(10))+[0]))

    def test_invert_indexes(self):
        # already tested in test_models.py
        pass

if __name__ == '__main__':
    unittest.main()
