# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for the models base class.

Its tests the abstract functionality provided by the Model class. It does not intend to test any specific implementation
of subclasses.

For generic testing of subclasses see test_models_generic.py.
"""

import unittest
import pandas as pd
from random import shuffle

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.mockup_model import MockUpModel
from mb_modelbase.models_core.mixable_cond_gaussian import MixableCondGaussianModel

# load data
from mb_modelbase.models_core.tests import test_crabs

# class TestDensity(unittest.TestCase):
#     """Test the model.probability method."""
#
#     def setUp(self):
#         # get 1d data
#         df = crabs.continuous().iloc[:, 0]
#
#         # learn model
#         model = MixableCondGaussianModel().fit(df)
#
#


class TestDataSelect(unittest.TestCase):
    """Test the model.select method."""

    def setUp(self):
        # crabs has columns: 'species', 'sex', 'FL', 'RW', 'CL', 'CW', 'BD'
        self.data = pd.DataFrame(_crabs_mixed)
        self.model = MockUpModel('crabs').set_data(_crabs_mixed)
        self.cols = list(_crabs_mixed.columns)
        self.shape = self.data.shape
        pass

    def test_it(self):
        result = self.model.select(what=['species'])
        self.assertTrue((self.shape[0],1) == result.shape
                        and result.columns[0] == 'species',
                        'tests that correct columns are returned')

        result = self.model.select(what=['FL','species','RW'])
        self.assertTrue((self.shape[0],3) == result.shape
                        and result.columns[0] == 'FL'
                        and result.columns[1] == 'species'
                        and result.columns[2] == 'RW',
                        'test that columns are in correct order')

        # test conditions are followed: no values incorrectly left out
        result = self.model.select(what=['FL','species','RW'])
        self.assertTrue((self.shape[0],3) == result.shape
                        and result.columns[0] == 'FL'
                        and result.columns[1] == 'species'
                        and result.columns[2] == 'RW',
                        'test that columns are in correct order')

        # test conditions are followed: no values incorrectly left in
        # TODO

        # test that invalid column names are reported as KeyErrors
        with self.assertRaises(KeyError):
            self.model.select(what=['foobar'])
        with self.assertRaises(KeyError):
            self.model.select(what=['foobar', 'RW'])


class TestInvalidParams(unittest.TestCase):

    def setUp(self):
        # setup model with known fixed paramters
        self.model = MockUpModel()
        self.model.generate_model(opts={'dim': 6})

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

    def test_subclasses(self):
        for class_ in TestAllModels.subclasses:
            self.assertTrue(issubclass(class_, Model))

    def test_automated_model_creation(self):
        for class_ in TestAllModels.subclasses:
            model = class_("foo")
            self.assertEqual(model.mode, 'empty')
            model.generate_model()
            self.assertEqual(model.mode, 'model')
            model._generate_data()
            self.assertEqual(model.mode, 'both')

if __name__ == '__main__':
    _crabs_mixed = test_crabs.mixed()
    _crabs_cat = test_crabs.continuous()
    _crabs_num = test_crabs.categorical()
    unittest.main()
