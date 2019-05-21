import numpy as np
import pandas as pd
import mb_modelbase as mbase
import unittest
from mb_modelbase.models_core.kde_model import KDEModel

class kde_test(unittest.TestCase):
    """
    Test the KDEModel
    """
    def test_fit_onedim(self):
        data = pd.DataFrame({'X': np.array([1, 2, 3, 3, 3, 4, 5])})
        kde_model = KDEModel()
        self.assertIsNone(kde_model.kde, "Before fitting there should be no kde object")
        kde_model.set_data(data)
        kde_model.fit()
        self.assertIsNotNone(kde_model.kde, "After fitting there should be a kde object")

    def test_fit_multidim(self):
        data = pd.DataFrame({'A': np.array([1, 2, 3, 3, 3, 4, 5]), 'B': np.array([1, 1, 1, 3, 3, 4, 5]),
                             'C': np.array([1, 2, 2, 2, 2, 2, 5]), 'D': np.array([2, 2, 3, 3, 3, 2, 5]),
                             'E': np.array([1, 5, 3, 3, 3, 5, 5]), 'F': np.array([1, 8, 3, 3, 5, 1, 5])})
        kde_model = KDEModel()
        self.assertIsNone(kde_model.kde, "Before fitting there should be no kde object")
        kde_model.set_data(data)
        kde_model.fit()
        self.assertIsNotNone(kde_model.kde, "After fitting there should be a kde object")

    # what should happen for categoricals?
    def test_fit_categoricals(self):
        data = pd.DataFrame({'A': np.array([1, 2, 3, 3, 3, 4, 5]), 'B': np.array(['1', '2', '3', '3', '3', '4', '5'])})
        kde_model = KDEModel()
        self.assertIsNone(kde_model.kde, "Before fitting there should be no kde object")
        kde_model.set_data(data)
        kde_model.fit()
        self.assertIsNotNone(kde_model.kde, "After fitting there should be a kde object")



