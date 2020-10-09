import unittest

import numpy as np
import pandas as pd

from mb.modelbase import MixableCondGaussianModel as MixCondGauss
from mb.modelbase.core.tests import test_allbus as ta


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.data = ta.mixed()
        self.model = MixCondGauss("TestMod")
        self.model.fit(df=self.data)
        pass


    def test_basics(self):
        self.assertEqual(self.model.name, 'TestMod')
        self.assertEqual(len(self.model._sample(5)), 5)

    def test_samplequality(self):

        samples1 = self.model._sample(250)
        data1 = pd.DataFrame(data=samples1, columns=self.model.names)
        testmod1 = MixCondGauss("Allbus_test1"); testmod1.fit(df=data1)
        error1 = ((np.array(testmod1._mu).ravel() - np.array(self.model._mu).ravel())**2).sum()

        samples2 = self.model._sample(500)
        data2 = pd.DataFrame(data=samples2, columns=self.model.names)
        testmod2 = MixCondGauss("Allbus_test2"); testmod2.fit(df=data2)
        error2 = ((np.array(testmod2._mu).ravel() - np.array(self.model._mu).ravel())**2).sum()

        self.assertTrue(error2 < error1)

    def test_onlycats(self):
        margmod = self.model.copy().marginalize(keep=['sex'])
        self.assertEqual(type(margmod._sample(1)[0][0]), type('str'))

    def test_onlynumericals(self):
        margmod = self.model.copy().marginalize(keep=['age'])
        self.assertEqual(type(margmod._sample(1)[0][0]), type(0.12345))

if __name__ == '__main__':
    unittest.main()