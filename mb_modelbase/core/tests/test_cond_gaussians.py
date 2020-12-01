# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for cond_gaussians.py
"""

import unittest
import numpy as np
import pandas as pd

from mb_modelbase.core.cond_gaussians import ConditionallyGaussianModel as CondGauss
from mb_modelbase.core.cond_gaussian.datasampling import genCGSample, genCatData, genCatDataJEx, cg_dummy


def print_info(model):
    # print some information about the model
    print(model)
    print('p_ML: \n', model._p)
    print('mu_ML: \n', model._mu)
    print('Sigma_ML: \n', model._S)


class TestJustRun(unittest.TestCase):
    """ This is not a real test case... it's just a couple of model queries that should go through
     without raising any exception """

    def test_A(self):
        # Sigma = np.diag([1, 1, 1])
        Sigma = np.matrix([[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])
        # Sigma = np.diag([1, 1, 1, 1])

        n = 1000
        testopts = {'levels': {0: [1, 2, 3, 4], 1: [2, 5, 10], 2: [1, 2]},
                    'Sigma': Sigma,
                    'fun': genCatData,
                    'catvalasmean': 1,  # works if dc = dg
                    'seed': 10}
        data = genCGSample(n, testopts)  # categoricals first, then gaussians, np array
        dc = len(testopts.keys())

        # fit model
        model = CondGauss('testmodel')
        model.fit(data)

        print_info(model)
        # ind = (0, 1, 1)
        # print('mu(', [model._extents[i][ind[i]] for i in ind], '):', model._mu[ind])
        # print(np.histogram(data[:, dc]))
        # plothist(data.iloc[:, dc + 1].ravel())
        pass

    def test_B(self):
        # Sigma = np.diag([1, 1, 1])
        # Sigma = np.matrix([[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])
        Sigma = np.diag([1, 1, 1, 1])

        n = 1000
        testopts = {'levels': {0: [0, 1], 1: [0, 1], 2: [0, 1]},
                    'Sigma': Sigma,
                    'fun': genCatDataJEx,
                    'catvalasmean': 1,
                    'seed': 10}
        data = genCGSample(n, testopts)  # categoricals first, then gaussians, np array
        dc = len(testopts.keys())
        print("dc ", dc)

        # fit model
        model = CondGauss('testmodel')
        model.fit(data)

        print_info(model)
        # ind = (0, 1, 1)
        # print('mu(', [model._extents[i][ind[i]] for i in ind], '):', model._mu[ind])
        # print(np.histogram(data[:, dc]))
        # plothist(data.iloc[:, dc + 1].ravel())
        pass

    def test_dummy_cg(self):
        # generate input data
        data = cg_dummy()

        # fit model
        model = CondGauss('testmodel')
        model.fit(data)

        print_info(model)
        print('p(M) = ', model._density(['M', 'Jena', 0, -6]))
        print('argmax of p(sex, city, age, income) = ', model._maximum())
        model.model(model=['sex', 'city', 'age'])  # marginalize income out
        print('p(M) = ', model._density(['M', 'Jena', 0]))
        print('argmax of p(sex, city, age) = ', model._maximum())
        model.model(model=['sex', 'age'], where=[('city', "==", 'Jena')])  # condition city out
        print('p(M) = ', model._density(['M', 0]))
        print('argmax of p(sex, agge) = ', model._maximum())
        model.model(model=['sex'], where=[('age', "==", 0)])  # condition age out
        print('p(M) = ', model._density(['M']))
        print('p(F) = ', model._density(['F']))
        print('argmax of p(sex) = ', model._maximum())


class TestInference(unittest.TestCase):

    def setUp(self):
        # setup model with known fixed paramters
        #m = CondGauss()
        #m._mu = ...
        pass

    def test_simple(self):
        # do all sorts of modelling queries and verify result by means of the parameters that are to be expected
        pass


class TestModelSelection(unittest.TestCase):

    def test_it(self):
        pass


if __name__ == '__main__':
    unittest.main()
