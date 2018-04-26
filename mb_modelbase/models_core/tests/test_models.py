# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for the models base class.

Its tests the abstract functionality provided by the Model class. It does not intend to test any specific implementation of subclasses.

For generic testing of subclasses see test_models_generic.py.
"""

import unittest
import pandas as pd
from random import shuffle

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.mockup_model import MockUpModel
from mb_modelbase.models_core.mixable_cond_gaussian import MixableCondGaussianModel
from mb_modelbase.models_core.tests import crabs


class TestDataSelect(unittest.TestCase):
    """Test the model.select method."""

    def setUp(self):
        _crabs_mixed = crabs.mixed()
        # crabs has columns: 'species', 'sex', 'FL', 'RW', 'CL', 'CW', 'BD'
        self.data = pd.DataFrame(_crabs_mixed)
        self.model = MockUpModel('crabs').set_data(_crabs_mixed)
        self.cols = list(_crabs_mixed.columns)
        self.shape = self.data.shape

    def test_it(self):
        result = self.model.select(what=['species'])
        result2 = self.model.select(what=['species'], data_category='test data')
        self.assertTrue(self.shape[0] == result.shape[0] + result2.shape[0], "tests that correct number of items is returned")

        self.assertTrue(result.columns[0] == result2.columns[0] == 'species'
                        and result.shape[1] == result2.shape[1] == 1, 'tests that correct columns are returned')

        result = self.model.select(what=['FL', 'species', 'RW'])
        self.assertTrue(3 == result.shape[1]
                        and result.columns[0] == 'FL'
                        and result.columns[1] == 'species'
                        and result.columns[2] == 'RW',
                        'test that columns are in correct order')

        # test conditions are followed: no values incorrectly left out
        result = self.model.select(what=['FL','species','RW'])
        self.assertTrue(3 == result.shape[1]
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
        # setup model with known fixed parameters
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


class TestDefaultValue(unittest.TestCase):
    """Test the default values."""

    """
    ## sample of crabs data frame
    species,sex,FL,RW,CL,CW,BD
    --------------------------------
    Blue,Male,8.1,6.7,16.1,19,7
    Blue,Male,8.8,7.7,18.1,20.8,7.4
    Blue,Male,9.2,7.8,19,22.4,7.7
    Orange,Female,10.7,9.7,21.4,24,9.8
    Orange,Female,11.4,9.2,21.7,24.1,9.7
    Orange,Female,12.5,10,24.1,27,10.9
    """

    @classmethod
    def setUpClass(cls):
        """Setup the model only once, since it is expensive and can be copied"""
        cls.data = crabs.mixed()
        cls.model = MixableCondGaussianModel('crabs').fit(cls.data)
        cls.model.mode = 'model'
        # crabs has columns: 'species', 'sex', 'FL', 'RW', 'CL', 'CW', 'BD'
        cls.cols = list(cls.data.columns)

    def setUp(self):
        pass

    def test_hide(self):
        m = __class__.model.copy()
        cols = __class__.cols

        self.assertEquals(m._hidden_count, 0, "tests initial hidden count.")
        self.assertEquals(m.hidden_fields(invert=True), cols, "tests initial Model.hidden_fields()")
        self.assertEquals(m.hidden_idxs(invert=True), list(range(len(cols))), "tests initial Model.hidden_fields()")

        # tests that cannot hide without default values
        for name in cols:
            with self.assertRaises(ValueError):
                m.hide(name)

        with self.assertRaises(ValueError):
            m.hide(cols)

        # hide some fields
        item = ['Orange', 'Female', 10.7, 9.7, 21.4, 24, 9.8]
        defaults = dict(zip(m.names, item))

        for names in [['sex'], ['sex', 'RW'], ['species', 'FL', 'CL'], m.names]:

            aggr_before = m.aggregate('maximum')

            # freeze (i.e. combined set_default() and hide())
            default_slice = {n: defaults[n] for n in names}
            m.freeze(default_slice)
            self.assertEqual(m._hidden_count, len(names), "tests hidden count.")
            self.assertEquals(set(m.hidden_fields()), set(names), "tests Model.hidden_fields()")
            aggr_after = m.aggregate('maximum')

            # determine alternative slice
            inv_names = m.inverse_names(names)
            aggr_before_slice = [v for k, v in dict(zip(m.names, aggr_before)).items() if k in inv_names]
            self.assertEqual(aggr_before_slice, aggr_after, 'tests invariance of aggregation under hide() and correct slicing of results aggregation')

            # unhide
            m.hide(names, False)
            self.assertEquals(m._hidden_count, 0, "tests unhiding.")
            aggr_after2 = m.aggregate('maximum')
            self.assertEqual(aggr_before, aggr_after2, 'tests invariance of aggregation under hide() unhide() cycle')

    def test_set_default(self):

        m = __class__.model.copy()

        def invariate_density(model, x, dims, x_alternative=None):
            """Given a model <model> a point (list) <x> and a list or single name of fields, it tests that
            the density doesn't change if dims are set to default values"""

            if isinstance(dims, str):  # make it a list
                dims = [dims]
            dims = m.sorted_names(dims)

            # check that value of density(x) does not change when we set a default value to field <dims>
            names = model.names

            x_before = dict(zip(names, x))  # x is given as list. we want a dict
            p_before = m.density(x_before)
            aggr_before = m.aggregate('maximum')

            x_defaults = {k: x_before[k] for k in dims}  # dict of defaults to set
            model.set_default(x_defaults)  # set defaults

            dims_inv = model.inverse_names(dims)
            x_after = {k: x_before[k] for k in dims_inv}  # generate new x without the values of <dims>
            p_after = m.density(x_after)
            aggr_after = m.aggregate('maximum')

            self.assertEqual(p_before, p_after, 'tests that value of density(x) does not change when we set a default value')

            self.assertEqual(aggr_before, aggr_after, 'tests that aggregation does not change when we set a default value')

            p_override = m.density(x_before)
            self.assertEqual(p_before, p_override, 'tests that overriding the default value with an identical value does not change the result')

            if x_alternative is not None:
                x_update = dict(zip(names, x_alternative))  # to dict
                x_update = {k: x_update[k] for k in dims}  # filter to the dims that we want to change
                x_alt = dict(x_before)
                x_alt.update(x_update)
                p_alt = m.density(x_alt)
                self.assertNotEqual(p_after, p_alt, 'tests that it results in a different density if we override the default value')

        item = ['Orange', 'Female', 10.7, 9.7, 21.4, 24, 9.8]  # items from the data set
        item2 = ['Blue', 'Male', 9.2, 7.8, 19, 22.4, 7.7]

        invariate_density(m, item, 'sex', item2)
        invariate_density(m, item, 'species', item2)
        invariate_density(m, item, 'FL', item2)
        invariate_density(m, item, ['sex', 'RW'], item2)
        invariate_density(m, item, ['CL', 'CW'], item2)
        # TODO: test other queries: aggregations, ...


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
            self.assertEqual(model.mode, None)
            model.generate_model()
            self.assertEqual(model.mode, 'model')
            model._generate_data()
            self.assertEqual(model.mode, 'both')


if __name__ == '__main__':
    print("test")
    unittest.main()