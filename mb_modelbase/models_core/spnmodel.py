#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:49:15 2018

@author: julien
@email: julien.klaus@uni-jena.de
"""

from mb_modelbase.models_core import Model
from mb_modelbase.models_core.spn.spn import SPNParams, SPN

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import copy as cp


class SPNModel(Model):
    def __init__(self, name, batchsize=1, mergebatch=10, \
                 corrthresh=0.1, equalweight=True, updatestruct=True, \
                 mvmaxscope=0, leaftype="normal", numcomp=2):
        super().__init__(name)
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        self.params = SPNParams(batchsize, mergebatch, corrthresh, \
                                equalweight, updatestruct, \
                                mvmaxscope, leaftype)
        self.numcomp = numcomp
        self.index = {}

    def _set_data(self, df, drop_silently=False):
      self._set_data_continuous(df, drop_silently)
      self.variables = len(self.fields)
      # creates a dict of name : index
      # looks strange but is correct
      self.nametoindex = dict((field["name"], i) for i,field in zip(range(len(self.fields)), self.fields))
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
      indexes = [self.nametoindex[i] for i in remove]
      for i in indexes:
         tmp[i] = True
      self.index.update(tmp)
      return []

    def _conditionout(self, keep, remove):
      tmp = {}
      indexes = [self.nametoindex[i] for i in remove]
      values = self._condition_values(remove)
      for (index,value) in zip(indexes, values):
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

    # calculated iterations times the maximum and returns the position 
    # with the highest densitiy value
    def _maximum(self, iterations=10):
        fun = lambda x :-1 * self._density(x)
        xlength = sum(1 for x in self.index.values() if x is None)
        xmax = None
        for i in range(iterations):
           x0 = np.random.randn(xlength)
           xopt = minimize(fun, x0, method='Nelder-Mead')
           if xmax is None or self._density(xmax) <=  self._density(xopt.x):
              xmax = xopt.x
        return xmax

    def copy(self, name=None):
      spncopy = super()._defaultcopy(name)
      
      spncopy._spnmodel = cp.deepcopy(self._spnmodel)
      spncopy.params = cp.deepcopy(self.params)
      spncopy.variables = self.variables
      spncopy.numcomp = self.numcomp
      spncopy.index = self.index.copy()
      spncopy.nametoindex = self.nametoindex.copy()
      return spncopy


if __name__ == "__main__":
    from sklearn.datasets import load_iris


    iris = load_iris()

    data = pd.DataFrame(data=np.c_[iris['data']],
                        columns=iris['feature_names'])

    spn = SPNModel("test")
    spn.set_data(data)
    spn._fit()
    x = spn.aggregate("maximum")


    