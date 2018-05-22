import unittest
import numpy as np

from mb_modelbase.models_core.mixable_cond_gaussian import MixableCondGaussianModel as MixCondGauss

# load data
from mb_modelbase.models_core.tests import test_allbus as ta

class TestMethods(unittest.TestCase):

    def setUp(self):
        self.data = ta.mixed()
        self.model = MixCondGauss("TestMod")
        self.model.fit(df=self.data)
        pass


    def test_something(self):
        self.assertEqual(self.model.name, 'TestMod')


if __name__ == '__main__':
    unittest.main()