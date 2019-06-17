import numpy as np
import pandas as pd
import mb_modelbase as mbase
import unittest
from mb_modelbase.models_core.kde_model import KDEModel
import math


class kde_test(unittest.TestCase):
    """
    Test the KDEModel
    """
    def test_fit_onedim(self):
        data = pd.DataFrame({'X': np.array([1, 2, 3, 3, 3, 4, 5])})
        kde_model = KDEModel('kde_model')
        self.assertIsNone(kde_model.kde, "Before fitting there should be no kde object")
        kde_model.set_data(data)
        kde_model.fit()
        self.assertIsNotNone(kde_model.kde, "After fitting there should be a kde object")

    def test_fit_multidim(self):
        data = pd.DataFrame({'A': np.array([1, 2, 3, 3, 3, 4, 5]), 'B': np.array([1, 1, 1, 3, 3, 4, 5]),
                             'C': np.array([1, 2, 2, 2, 2, 2, 5]), 'D': np.array([2, 2, 3, 3, 3, 2, 5]),
                             'E': np.array([1, 5, 3, 3, 3, 5, 5]), 'F': np.array([1, 8, 3, 3, 5, 1, 5])})
        kde_model = KDEModel('kde_model')
        self.assertIsNone(kde_model.kde, "Before fitting there should be no kde object")
        kde_model.set_data(data)
        kde_model.fit()
        self.assertIsNotNone(kde_model.kde, "After fitting there should be a kde object")

    def test_conditionout(self):
        data = pd.DataFrame({'B': np.array([2, 4, 7, 7, 7, 4, 1]), 'A': np.array([1, 2, 3, 3, 3, 4, 5])},
                            columns=['B', 'A'])
        kde_model = KDEModel('kde_model')
        kde_model.fit(data)
        self.assertTrue(kde_model.data.sort_values(by='A').reset_index(drop=True).equals(data),
                        "input data was not passed properly to the model")
        # Change domains of dimension A
        kde_model.fields[1]['domain'].setlowerbound(2)
        kde_model.fields[1]['domain'].setupperbound(4)
        # Condition and marginalize model
        kde_model._conditionout(keep='B', remove='A')
        # Generate control data
        data_cond = pd.DataFrame({'B': np.array([4, 7, 7, 7, 4]), 'A': np.array([2, 3, 3, 3, 4])},
                                 columns=['B', 'A'])
        self.assertTrue(kde_model.data.sort_values(by='A').reset_index(drop=True).equals(data_cond),
                        "model data was not marginalized and conditioned properly")

    def test_maximum(self):
        data = pd.DataFrame({'B': np.array([0, 2, 2, 3, 3, 3, 4, 4, 6]), 'A': np.array([1, 2, 3, 3, 3, 3, 3, 4, 5])},
                            columns=['B', 'A'])
        maximum = np.array([3., 3.])
        kde_model = KDEModel('kde_model')
        kde_model.fit(data)
        model_max = kde_model._maximum()

        for i in range(len(kde_model.fields)):
            self.assertAlmostEqual(model_max[i], maximum[i])

    def test_predict(self):
        data = pd.DataFrame({'A': np.array([1, 2, 3, 2, 3, 4, 5]),
                             'B': np.array([1, 2, 2, 3, 3, 4, 5]),
                             'C': np.array([1, 2, 3, 3, 3, 4, 5]),
                             'D': np.array([0, 3, 3, 4, 4, 6, 7])},
                            columns=['A', 'B', 'C', 'D'])
        kde_model = KDEModel('kde_model')
        kde_model.fit(data)
        # marginalize out all but two dimensions
        kde_model.marginalize(keep=['C', 'D'])
        # condition out one of the remaining dimensions
        kde_model.byname('C')['domain'].setupperbound(3)
        kde_model.byname('C')['domain'].setlowerbound(2)
        kde_model.marginalize(keep=['D'])
        # For the remaining dimension: get point of maximum/average probability density
        self.assertAlmostEqual(kde_model._maximum(), np.array([3.0]), 'prediction is not correct')


def test_mixed_categorical_numerical_model(self):
    data = pd.DataFrame({'A': np.array([1, 2, 3, 2, 3, 4, 5]),
                         'B': np.array(['foo', 'bar', 'foo', 'foo', 'bar', 'foo', 'bar'])},
                        columns=['A', 'B'])
    kde_model = KDEModel('kde_model')
    kde_model.fit(data)
    self.assertAlmostEqual(kde_model._density([1, 'foo']), 0.4, 'density is not calculated correctly')

    # def test_discrete_domains(self):
    #     data = pd.DataFrame({'A': np.array([1, 2, 3, 3, 3, 4, 5]), 'B': np.array(['1', '2', '3', '3', '3', '4', '5'])})
    #     kde_model = KDEModel('kde_model')
    #     self.assertIsNone(kde_model.kde, "Before fitting there should be no kde object")
    #     kde_model.set_data(data)
    #     kde_model.fit()
    #     kde_model.byname('B')['domain']._value = ['2', '4']
    #     kde_model.marginalize(keep=['A'])
    #     self.assertEqual(kde_model._arithmetic_mean(), 3.0, 'prediction is not correct')

