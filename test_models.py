# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for cond_gaussians.py
"""

import unittest
import numpy as np
import pandas as pd
from random import shuffle

from models import Model
from mockup_model import MockUpModel
from categoricals import CategoricalModel
from gaussians import MultiVariateGaussianModel as GaussianModel
from cond_gaussians import ConditionallyGaussianModel as CGModel


class TestJustRun(unittest.TestCase):
    """ This is not a real test case... it's just a couple of model queries that should go through
     without raising any exception """

    def test_A(self):
        pass


class TestInvalidParams(unittest.TestCase):

    def setUp(self):
        # setup model with known fixed paramters
        self.model = MockUpModel()
        self.model._generate_model(opts={'dim': 6})

    def test_asindex(self):
        c = self.model
        self.assertEqual(1, c.asindex('dim1'))
        self.assertEqual([1, 2, 3], c.asindex(['dim1', 'dim2', 'dim3']))
        self.assertEqual([2, 1, 3], c.asindex(['dim2', 'dim1', 'dim3']))
        with self.assertRaises(KeyError):
            c.asindex('foo')
        with self.assertRaises(KeyError):
            c.asindex(['foo'])
        with self.assertRaises(KeyError):
            c.asindex(['dim1', 'foo'])

    def test_byname(self):
        m = self.model
        f = m.fields
        self.assertEqual(f[1], m.byname('dim1'))
        self.assertEqual([f[1], f[2], f[3]], m.byname(['dim1', 'dim2', 'dim3']))
        self.assertEqual([f[2], f[1], f[3]], m.byname(['dim2', 'dim1', 'dim3']))
        with self.assertRaises(KeyError):
            m.byname('foo')
        with self.assertRaises(KeyError):
            m.byname(['foo'])
        with self.assertRaises(KeyError):
            m.byname(['dim1', 'foo'])

    def test_isfieldname(self):
        m = self.model
        self.assertTrue(m.isfieldname('dim1'))
        self.assertTrue(m.isfieldname(['dim1', 'dim2', 'dim3']))
        self.assertTrue(m.isfieldname(['dim2', 'dim1', 'dim3']))
        self.assertFalse(m.isfieldname('foo'))
        self.assertFalse(m.isfieldname(['foo']))
        self.assertFalse(m.isfieldname(['dim1', 'foo']))

    def test_inverse_names(self):
        m = self.model
        self.assertEqual(m.names, ['dim0', 'dim1', 'dim2', 'dim3', 'dim4', 'dim5'],
                         'the test after this test require that its a 6d model.')

        self.assertEqual(['dim1', 'dim2', 'dim3', 'dim4', 'dim5'], m.inverse_names('dim0'))
        self.assertEqual(['dim0', 'dim2', 'dim3', 'dim4', 'dim5'], m.inverse_names('dim1'))
        self.assertEqual(['dim0', 'dim1', 'dim2', 'dim3', 'dim4'], m.inverse_names('dim5'))
        self.assertEqual(['dim0', 'dim1', 'dim2', 'dim3', 'dim4', 'dim5'], m.inverse_names('foo'))

        self.assertEqual(['dim1', 'dim2', 'dim3', 'dim4', 'dim5'], m.inverse_names(['dim0']))
        self.assertEqual(['dim2', 'dim3', 'dim4', 'dim5'], m.inverse_names(['dim0', 'dim1']))
        self.assertEqual(['dim1', 'dim2', 'dim3', 'dim4'], m.inverse_names(['dim5', 'dim0']))
        self.assertEqual(['dim1', 'dim2', 'dim3', 'dim4'], m.inverse_names(['bar', 'dim5', 'dim0', 'foo']))
        self.assertEqual(['dim0', 'dim1', 'dim2', 'dim3', 'dim4', 'dim5'], m.inverse_names(['foo', 'bar']))
        self.assertEqual([], m.inverse_names(['dim0', 'dim1', 'dim2', 'dim3', 'dim4', 'dim5']))

    def test_sorted_names(self):
        m = self.model
        self.assertEqual(m.names, ['dim0', 'dim1', 'dim2', 'dim3', 'dim4', 'dim5'],
                         'the test after this test require that its a 6d model.')

        self.assertEqual(m.names, m.sorted_names(['dim0', 'dim1', 'dim2', 'dim3', 'dim4', 'dim5']))
        self.assertEqual([], m.sorted_names([]))
        for i in range(20):
            shuffled = list(m.names)
            shuffle(shuffled)
            self.assertEqual(m.names, m.sorted_names(shuffled))

        for i in range(20):
            shuffled = list(m.names) + ['foo', 'bar', 'foobar']
            shuffle(shuffled)
            self.assertEqual(m.names, m.sorted_names(shuffled))


class TestAllModels(unittest.TestCase):
    """This test case tests simple invariants that should hold for all concrete models, that however,
    are not abstracted into the abstract base class."""

    # register all subclasses here
    subclasses = [MockUpModel]
    # subclasses = [MockUpModel, CategoricalModel, GaussianModel, CGModel]

    def test_subclasses(self):
        for class_ in TestAllModels.subclasses:
            self.assertTrue(issubclass(class_, Model))

    def test_automated_model_creation(self):
        for class_ in TestAllModels.subclasses:
            model = class_("foo")
            self.assertEqual(model._mode, 'empty')
            model._generate_model()
            self.assertEqual(model._mode, 'model')
            model._generate_data()
            self.assertEqual(model._mode, 'both')

if __name__ == '__main__':
    unittest.main()
