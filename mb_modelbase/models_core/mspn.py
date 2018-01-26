#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:48:00 2018

@author: julien
@email: julien.klaus@uni-jena.de
"""
from mb_modelbase.models_core import Model

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import copy as cp


class mspn(Model):
    def __init__(self, name):
       super().__init__(name)

    def _set_data(self, df, drop_silently=False):
       raise NotImplemented()
       return []

    def _fit(self, iterations=5):
       raise NotImplemented()
       return []

    def _marginalizeout(self, keep, remove):
       raise NotImplemented()
       return []

    def _conditionout(self, keep, remove):
       raise NotImplemented()
       return []

    def _density(self, x):
        raise NotImplemented()

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

      raise NotImplemented()

      return spncopy


if __name__ == "__main__":
    from sklearn.datasets import load_iris


    iris = load_iris()

    data = pd.DataFrame(data=np.c_[iris['data']],
                        columns=iris['feature_names'])



