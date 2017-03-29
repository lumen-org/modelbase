# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for cond_gaussians_wm.py
"""

import unittest
from cond_gaussian_wm import CgWmModel
from cond_gaussian.datasampling import cg_dummy


def print_info(model):
    # print some information about the model
    print(model)
    print('p_ML: \n', model._p)
    print('mu_ML: \n', model._mu)
    print('Sigma_ML: \n', model._S)


class TestJustRun(unittest.TestCase):
    """ This is not a real test case... it's just a couple of model queries that should go through
     without raising any exception """

    def test_dummy_cg(self):
        # generate input data
        data = cg_dummy()

        # fit model
        model = CgWmModel('testmodel')
        model.fit(data)
        original = model.copy()

        # marginalize a single continuous variable out: income
        model = original.copy().model(model=['sex', 'city', 'age'])
        print("model of sex, city, age:\n", model)

        # marginalize two continuous variables out: income, age
        model = original.copy().model(model=['sex', 'city'])
        print("model of sex, city:\n", model)

        # marginalize a single categorical variable out: sex
        model = original.copy().model(model=['city', 'age', 'income'])
        print("model of city, age, income:\n", model)

        # marginalize two categorical variables out: sex, city
        model = original.copy().model(model=['age', 'income'])
        print("model of age, income:\n", model)

        # marginalize a continuous and a categorical variable out: sex, income
        model = original.copy().model(model=['city', 'age'])
        print("model of city, age:\n", model)

        # marginalize two continuous and a single categorical variable out: sex, income, age
        model = original.copy().model(model=['city'])
        print("model of city:\n", model)

        # marginalize a single continuous and a two categorical variables out: sex, city, age
        model = original.copy().model(model=['income'])
        print("model of income:\n", model)

        ## older stuff
        model = original.copy()
        print('p(M) = ', model._density(['M', 'Jena', 0, -6]))

        print('argmax of p(sex, city, age, income) = ', model._maximum())

        print('p(M) = ', model._density(['M', 'Jena', 0]))

        print('argmax of p(sex, city, age) = ', model._maximum())

        model.model(model=['sex', 'age'], where=[('city', "==", 'Jena')])  # condition city out
        print("model [sex, city == Jena, age]:\n", model)

        print('p(M) = ', model._density(['M', 0]))

        print('argmax of p(sex, agge) = ', model._maximum())

        model.model(model=['sex'], where=[('age', "==", 0)])  # condition age out
        print("model [sex, city == Jena, age == 0]:\n", model)

        print('p(M) = ', model._density(['M']))

        print('p(F) = ', model._density(['F']))

        print('argmax of p(sex) = ', model._maximum())

        model = original.copy().model(['sex', 'age', 'income'])  # marginalize city out
        print("model [sex, age, income]:\n", model)


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
