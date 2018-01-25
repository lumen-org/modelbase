# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for gaussians.py
"""

import unittest
import numpy.testing as npt
import numpy as np

from mb_modelbase.models_core.gaussians import MultiVariateGaussianModel as Gaussian


class TestRunningShouldNotRaiseException(unittest.TestCase):
    """ This is not a real test case... it's just a couple of model queries that should go through
     without raising any exception """

    def test_1(self):
        sigma = np.matrix([
            [1.0, 0.6, 0.0, 2.0],
            [0.6, 1.0, 0.4, 0.0],
            [0.0, 0.4, 1.0, 0.0],
            [2.0, 0.0, 0.0, 1.]])
        mu = np.matrix([1.0, 2.0, 0.0, 0.5]).T
        foo = Gaussian("foo")
        opts = {'mode': 'custom', 'sigma': sigma, 'mu': mu}
        foo.generate_model(opts)
        foocp = foo.copy("foocp")
        print("\n\nmodel 1\n" + str(foocp))
        foocp2 = foocp.model(['dim1', 'dim0'], as_="foocp2")
        print("\n\nmodel 2\n" + str(foocp2))

        res = foo.predict(predict=['dim0'], splitby=[('dim0', 'equidist', [5])])
        print("\n\npredict 1\n" + str(res))

        res = foo.predict(predict=[(['dim1'], 'maximum', 'dim1', []), 'dim0'],
                          splitby=[('dim0', 'equidist', [10])])
        print("\n\npredict 2\n" + str(res))

        res = foo.predict(predict=[(['dim0'], 'maximum', 'dim0', []), 'dim0'],
                          where=[('dim0', 'equals', 1)], splitby=[('dim0', 'equidist', [10])])
        print("\n\npredict 3\n" + str(res))

        res = foo.predict(predict=[(['dim0'], 'density', 'dim0', []), 'dim0'],
                           splitby=[('dim0', 'equidist', [10])])
        print("\n\npredict 4\n" + str(res))

        res = foo.predict(
            predict=[(['dim0'], 'density', 'dim0', []), 'dim0'],
            splitby=[('dim0', 'equidist', [10])],
            where=[('dim0', 'greater', -1)])
        print("\n\npredict 5\n" + str(res))

        res = foo.predict(
            predict=[(['dim0'], 'density', 'dim0', []), 'dim0'],
            splitby=[('dim0', 'equidist', [10])],
            where=[('dim0', 'less', -1)])
        print("\n\npredict 6\n" + str(res))

        res = foo.predict(
            predict=[(['dim0'], 'density', 'dim0', []), 'dim0'],
            splitby=[('dim0', 'equidist', [10])],
            where=[('dim0', 'less', 0), ('dim2', 'equals', -5.0)])
        print("\n\npredict 7\n" + str(res))

        res, base = foo.predict(
            predict=[(['dim0'], 'density', 'dim0', []), 'dim0'],
            splitby=[('dim0', 'equidist', [10]), ('dim1', 'equidist', [7])],
            where=[('dim0', 'less', -1), ('dim2', 'equals', -5.0)],
            returnbasemodel=True)
        print("\n\npredict 8\n" + str(res))

        res, base = foo.predict(
            predict=[(['dim0'], 'average', 'dim0', []), (['dim0'], 'density', 'dim0', []),
                     'dim0'],
            splitby=[('dim0', 'equidist', [10])],
            # where=[('dim0', 'less', -1), ('dim2', 'equals', -5.0)],
            returnbasemodel=True)
        print("\n\npredict 9\n" + str(res))

        res, base = foo.predict(
            predict=['dim0', 'dim1', (['dim0', 'dim1'], 'average', 'dim0', []),
                     (['dim0', 'dim1'], 'average', 'dim1', [])],
            splitby=[('dim0', 'identity', []), ('dim1', 'equidist', [4])],
            where=[('dim0', '<', 2), ('dim0', '>', 1)],
            returnbasemodel=True)
        print("\n\npredict 10\n" + str(res))


class TestInferenceLvl01(unittest.TestCase):
    
    def setUp(self):
        # setup model with zero mean and identity matrix as sigma
        self.m = Gaussian("test")
        self.m.generate_model(opts={'mode': 'normal', 'dim': 5})

    def test_marginalize_1(self):
        m = self.m
        m.model(model=m.names[1:4])
        dim = m.dim
        npt.assert_equal(m._S, np.eye(dim))
        npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)

        m.model(model=m.names[0:2])
        dim = m.dim
        npt.assert_equal(m._S, np.eye(dim))
        npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)

    def test_condition_1(self):
        m = self.m
        m.model(model='*', where=[('dim1', '==', 0)])
        dim = m.dim
        npt.assert_equal(m._S, np.eye(dim))
        npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)

        m.model(model='*', where=[('dim3', '==', 0), ('dim4', '==', 0)])
        dim = m.dim
        npt.assert_equal(m._S, np.eye(dim))
        npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)

    def test_maximum_1(self):
        m = self.m
        res = m.aggregate('maximum')
        npt.assert_equal(res, np.zeros(m.dim))

    def test_predict_1(self):
        m = self.m
        res = m.aggregate('maximum')
        npt.assert_equal(res, np.zeros(m.dim))
        

class TestSimple2DMVG(unittest.TestCase):

    def setUp(self):
        self.m = Gaussian("test")
        mu = np.matrix(np.zeros(2)).T
        sigma = np.matrix([[1, 0.5], [0.5, 1]])
        self.m.generate_model({'mode': 'custom', 'mu': mu, 'sigma': sigma})
        self.m._generate_data({'n': 500})

    def test_it(self):
        m = self.m
        result = m.predict(predict=[(['dim0'], 'maximum', 'dim0', [])], where=[('dim1', '==', 1)])
        print(result)
        # self.assertEqual(result, 0.5)
        result = m.predict(predict=['dim1', (['dim0'], 'maximum', 'dim0', [])], splitby=[('dim1', 'equidist', [5])])
        print(result)
        result = m.predict(predict=['dim1', (['dim0'], 'maximum', 'dim0', [])], splitby=[('dim1', 'equiinterval', [5])])
        print(result)


if __name__ == '__main__':
    unittest.main()
