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
from mb_modelbase.models_core.models import Density


class TestRunningShouldNotRaiseException(unittest.TestCase):
    """ This is not a real test case... it's just a couple of model queries that should go through
     without raising any exception """

    def test_1(self):
        df = pd.read_csv('categorical_dummy.csv')
        model = CategoricalModel('test')
        model.fit(df)

        print('model:', model)
        print('model._p:', model._p)

        res = model.predict(
            predict=['Student', 'City', 'Sex', Density(['City', 'Sex', 'Student'])],
            splitby=[('City', 'elements', []), ('Sex', 'elements', []), ('Student', 'elements', [])])
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


if __name__ == '__main__':
    unittest.main()

