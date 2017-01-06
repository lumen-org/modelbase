# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for cond_gaussians.py
"""

import unittest
import models as model  # load models using model.load
from cond_gaussians import ConditionallyGaussianModel as CondGauss

class TestDemo(unittest.TestCase):

    def test_it(self):
        self.assertEqual("0", "0")

class TestInference(unittest.TestCase):

    def setUp(self):
        # setup model with known fixed paramters
        m = CondGauss()
        m._mu = ...  #TODO

    def test_simple(self):
        # do all sorts of modelling queries and verify result by means of the parameters that are to be expected


class TestModelSelection(unittest.TestCase):

    def test_it(self):
        pass


if __name__ == '__main__':
    unittest.main()
