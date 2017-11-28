# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for the modelbase module.py
"""

import unittest

from modelbase import ModelBase
import data.crabs.crabs as crabs


class TestJustRun(unittest.TestCase):
    """ This is not a real test case... it's just a couple of model queries that should go through
     without raising any exception """

    def setUp(self):
        self.mb = ModelBase(name="my mb", model_dir='test_models')

    def test_A(self):
        self.assertTrue('cg_crabs' in self.mb.list_models(), 'this test suite requires the cg_crabs model')

        # crabs has columns: 'species', 'sex', 'FL', 'RW', 'CL', 'CW', 'BD'
        # now run a few queries
        result = self.mb.execute('{"SELECT": ["sex", "FL"], "FROM": "cg_crabs"}')
