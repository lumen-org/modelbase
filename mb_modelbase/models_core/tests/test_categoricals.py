# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Test Suite for categoricals.py
"""

import unittest
import numpy.testing as npt
import numpy as np
import pandas as pd

from mb_modelbase.models_core.categoricals import CategoricalModel as CategoricalModel


class TestRunningShouldNotRaiseException(unittest.TestCase):
    """ This is not a real test case... it's just a couple of model queries that should go through
     without raising any exception """

    def test_1(self):
        df = pd.read_csv('data/categorical_dummy.csv')
        model = CategoricalModel('test')
        model.fit(df)

        print('model:', model)
        print('model._p:', model._p)

        res = model.predict(
            predict=['Student', 'City', 'Sex',
                    (['City', 'Sex', 'Student'], 'density', 'density', [])],
            splitby=[('City', 'elements', []),('Sex', 'elements', []),
                    ('Student', 'elements', [])])
        print('probability table: \n', str(res))

        res = model.predict(
            predict=['City', 'Sex',
                    (['City', 'Sex'], 'density', 'density', [])],
            splitby=[('City', 'elements', []),('Sex', 'elements', [])])
        print("\n\npredict marginal table: city, sex\n" + str(res))

        res = model.predict(
            predict=['City',
                    (['Sex'], 'maximum', 'Sex', [])],
            splitby=[('City', 'elements', [])])
        print("\n\npredict most likely sex by city: \n" + str(res))

        res = model.predict(
            predict=['Sex',
                    (['Sex'], 'density', 'density', [])],
            splitby=[('Sex', 'elements', [])])
        print("\n\npredict marginal table: sex\n" + str(res))

        res = model.predict(
            predict=['City', 'Sex',
                    (['City', 'Sex'], 'density', 'density', [])],
            splitby=[('City', 'elements', []),('Sex', 'elements', [])],
            where=[('Student', '==', 'yes')])
        print("\n\nconditional prop table: p(city, sex| student=T):\n" + str(res))

        print('\n\nafter this comes less organized output:')
        print('model density 1:', model._density(['Jena', 'M', 'no']))
        print('model density 1:', model._density(['Erfurt', 'F', 'yes']))

        print('model maximum:', model._maximum())

        marginalAB = model.copy().marginalize(keep=['City', 'Sex'])
        print('marginal on City, Sex:', marginalAB)
        print('marginal p: ', marginalAB._p)

        conditionalB = marginalAB.condition([('City', '==', 'Jena')]).marginalize(keep=['Sex'])
        print('conditional Sex|City = Jena: ', conditionalB)
        print('conditional Sex|City = Jena: ', conditionalB._p)

        print('most probable city: ', model.copy().marginalize(keep=['City']).aggregate('maximum'))
        print('most probably gender in Jena: ',
              model.copy().condition([('City', '==', 'Jena')]).marginalize(keep=['Sex']).aggregate('maximum'))
        print('most probably Sex for Students in Erfurt: ',
              model.copy().condition([('City', '==', 'Erfurt'), ('Student', '==', 'yes')]).marginalize(keep=['Sex']).aggregate(
                  'maximum'))


# class TestInferenceLvl01(unittest.TestCase):
#     def setUp(self):
#         # setup model with zero mean and identiy matrix as sigma
#         self.m = Gaussian("test")
#         self.m._generate_model(opts={'mode': 'normal', 'dim': 5})
#
#     def test_marginalize_1(self):
#         m = self.m
#         m.model(model=m.names[1:4])
#         dim = m.dim
#         npt.assert_equal(m._S, np.eye(dim))
#         npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)
#
#         m.model(model=m.names[0:2])
#         dim = m.dim
#         npt.assert_equal(m._S, np.eye(dim))
#         npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)
#
#     def test_condition_1(self):
#         m = self.m
#         m.model(model='*', where=[('dim1', '==', 0)])
#         dim = m.dim
#         npt.assert_equal(m._S, np.eye(dim))
#         npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)
#
#         m.model(model='*', where=[('dim3', '==', 0), ('dim4', '==', 0)])
#         dim = m.dim
#         npt.assert_equal(m._S, np.eye(dim))
#         npt.assert_equal(m._mu, np.matrix(np.zeros(dim)).T)
#
#     def test_maximum_1(self):
#         m = self.m
#         res = m.aggregate('maximum')
#         npt.assert_equal(res, np.zeros(m.dim))
#
#     def test_predict_1(self):
#         m = self.m
#         res = m.aggregate('maximum')
#         npt.assert_equal(res, np.zeros(m.dim))


if __name__ == '__main__':
    unittest.main()

