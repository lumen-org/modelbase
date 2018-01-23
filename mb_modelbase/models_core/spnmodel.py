#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:49:15 2018

@author: julien
@email: julien.klaus@uni-jena.de
"""

from mb_modelbase.models_core import Model
from mb_modelbase.models_core.spn.spn import SPNParams, SPN
import copy as cp

import numpy as np
import pandas as pd


class SPNModel(Model):
    def __init__(self, name, batchsize=1, mergebatch=10, \
                 corrthresh=0.1, equalweight=True, updatestruct=True, \
                 mvmaxscope=0, leaftype="normal", numcomp=2):
        super().__init__(name)
        self.params = SPNParams(batchsize, mergebatch, corrthresh, \
                                equalweight, updatestruct, \
                                mvmaxscope, leaftype)
        self.numcomp = numcomp
        self.index = {}

    def _set_data(self, df, drop_silently=False):
        self._set_data_continuous(df, drop_silently)
        self.variables = len(self.fields)
        for i in range(self.variables):
            self.index[i] = None
        return []

    def _fit(self, iterations=5):
        data = self.data.get_values()
        if self.data.empty:
            raise Exception("No data available to fit on.")
        self._spnmodel = SPN(self.variables, self.numcomp, self.params)
        for i in range(iterations):
            self._spnmodel.update(data)
        return []

    def _marginalizeout(self, keep, remove):
        tmp = {}
        remove = self.asindex(remove)
        for i in remove:
            tmp[i] = True
        self.index.update(tmp)
        return []

    def _conditionout(self, keep, remove):
        tmp = {}
        indexes = self.asindex(remove)
        values = self._condition_values(remove)
        for (index, value) in zip(indexes, values):
            tmp[index] = value
        self.index.update(tmp)
        return []

    def _density(self, x):
        j = 0
        tmp = self.index.copy()
        for i in tmp.keys():
            if tmp[i] is None:
                tmp[i] = x[j]
                j += 1
        if len(x) != j:
            raise Exception("Two many values.")
        return np.exp(self._spnmodel.evaluate(None, tmp))[0]

    def copy(self, name=None):
        spncopy = self._defaultcopy(name)
        spncopy._spnmodel = cp.deepcopy(self._spnmodel)
        spncopy.index = self.index.copy()
        spncopy.params = cp.deepcopy(self.params)
        return spncopy


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from mb_modelbase.models_core.spn import generateSPNPdf

    iris = load_iris()

    data = pd.DataFrame(data=np.c_[iris['data']],
                        columns=iris['feature_names'])

    spn = SPNModel("test")

    spn._set_data(data)
    spn._fit(iterations=0)
    generateSPNPdf(spn._spnmodel, filename="img/test1")
    spn2 = spn.copy()
    spn2._fit(iterations=1)
    print(spn2.index)
    spn3 = spn2._marginalizeout([0], [1,2,3])
    print(spn3.index)
    spn4 = spn3._conditionout([(1, 1.5)], [0, 2, 3])
    print(spn4.index)
    print(spn4._density([4.5, 1.8]))
    generateSPNPdf(spn._spnmodel, filename="img/test3")
    generateSPNPdf(spn2._spnmodel, filename="img/test2")
