#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:48:00 2018

@author: julien
@email: julien.klaus@uni-jena.de

Note to get it run:

In addition to any dependencies required by the mb_modelbase package you also need:
    (1) these python packages available:
        rpy2, joblib, tensorflow, networkx, numba, mpmath
      to install pip install rpy2, joblib, tensorflow, networkx, numba, mpmath

    (2) R installed, with these packages available:
        histogram, iterators, partykit, foreach, doMC, dplyr, gtools, igraph, energy, dummies
      to install, run from an R console:
        install.packages(c("digest", "histogram", "iterators", "partykit", "foreach", "doMC", "dplyr", "gtools", "igraph", "energy", "dummies"))

What did we do to make it run:
  * in particular we changed line ~ tfspn.py:1230
  * we pushed our index through the eval. if the index of the
    RV is True then we return simply 1.0 for the marginalization
    
IMPORTANT:
  * the model is not normalized
  * to normalize it you have to marginalize out all RV and then get the density, which is your normalisation factor
"""

import rpy2

from mb_modelbase.models_core import Model, get_columns_by_dtype, to_category_cols

from mb_modelbase.models_core.mspn.tfspn.piecewise import estimate_domains
from mb_modelbase.models_core.mspn.tfspn.SPN import SPN, Splitting

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import copy as cp
from collections import Iterable


class MSPNModel(Model):
    def __init__(self, name, threshold=None, min_instances_slice=None):
        super().__init__(name)
        self.featureTypes = None
        self.threshold = threshold if threshold is not None else 0.4
        self.min_instances_slice = min_instances_slice
        self.index = {}
        self.normalizeFactor = 1.0
        self._aggrMethods = {
            'maximum': self._maximum,
        }

    def _set_data(self, df, drop_silently=False):
        _, cat_names, num_names = get_columns_by_dtype(df)
        df = to_category_cols(df, cat_names)
        self._set_data_mixed(df, drop_silently, num_names, cat_names)

        self.featureTypes = ["categorical"]*len(cat_names) + ['continuous']*len(num_names)
        if self.min_instances_slice is None:
            # if there is not minimum slice width we set it to 8% of the data length
            self.min_instances_slice = len(self.data) * 0.08
        self.nametoindex = dict((field["name"], i) for i, field in zip(range(len(self.fields)), self.fields))
        for i in range(len(self.fields)):
            self.index[i] = None
        return []

    def _fit(self):
        # data = np.array(self.data)
        # for i in range(len(data)):
        #     data[i] = [int(j) if self.featureTypes[i] == "categorical" else j for j in data[i]]
        self._mspnmodel = SPN.LearnStructure(np.array(self.data), featureTypes=self.featureTypes, \
                                             row_split_method=Splitting.KmeansRows(), \
                                             col_split_method=Splitting.RDCTest(threshold=self.threshold), \
                                             min_instances_slice=self.min_instances_slice).root
        self.normalizeFactor = self._getNormalizeFactor()
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
        for (index, value) in zip(indexes, values):
            tmp[index] = value
        self.index.update(tmp)
        return []

    def _density(self, x):

        x = [float(e) for e in x]

        j = 0
        tmp = self.index.copy()
        for i in tmp.keys():
            if tmp[i] is None:
                tmp[i] = x[j]
                j += 1
        if len(x) != j:
            raise Exception("Two many values.")
        return np.exp(self._mspnmodel.eval(None, tmp))[0]/self.normalizeFactor

    # calculated iterations times the maximum and returns the position
    # with the highest densitiy value
    def _maximum(self,steps=3):
        fun = lambda x: -1 * self._density(x)
        xmax = None
        #there should be a lot of start vectors
        for x0 in self._getStartVectors(steps):
            #print(x0)
            xopt = minimize(fun, x0, method='Nelder-Mead')
            if xmax is None or self._density(xmax) <= self._density(xopt.x):
                xmax = xopt.x
        return xmax

    def copy(self, name=None):
        spncopy = super()._defaultcopy(name)
        spncopy.featureTypes = self.featureTypes
        spncopy.threshold = self.threshold
        spncopy.min_instance_slice = self.min_instances_slice
        # spncopy._mspnmodel = cp.deepcopy(self._mspnmodel)
        spncopy._mspnmodel = self._mspnmodel
        spncopy.nametoindex = self.nametoindex.copy()
        spncopy.index = self.index.copy()
        return spncopy
     
    def _getDomains(self):
       w=[self._mspnmodel]
       domains = dict()
       while len(w) > 0:
          n = w.pop(0)
          if n.__class__.__name__ is "PiecewiseLinearPDFNode":
             if n.featureIdx in domains.keys():
                if len(domains[n.featureIdx]) > 1: 
                   domains[n.featureIdx][0] = n.domain[0] if domains[n.featureIdx][0] < n.domain[0] else domains[n.featureIdx][0]
                if len(domains[n.featureIdx]) > 1: 
                   domains[n.featureIdx][-1] = n.domain[-1] if domains[n.featureIdx][-1] > n.domain[-1] else domains[n.featureIdx][-1] 
             else: 
                domains[n.featureIdx] = n.domain
          for c in n.children:
             w.append(c)
       return domains
     
    def _getRandomVector(self):
       domains = self._getDomains()
       xlength = sum(1 for x in self.index.values() if x is None)
       x0 = np.zeros(xlength)
       j = 0
       for i in self.index.keys():
            if self.index[i] is None:
                x0[j] = np.random.uniform(domains[i][0],domains[i][-1])
                j += 1
       return x0
    
    def _getNormalizeFactor(self):
        #copy the current index
        tmp = self.index.copy()
        #lets normalize, through this key, eval allways returns 1.0
        self.index["norm"] = True
        xlength = sum([1 for i in self.index.values() if i is None or i is True])
        #it is not important what x is, only the length (because of norm -1)
        normalizeFactor = self._density([1.0]*(xlength-1))
        #reset the index
        self.index = tmp
        return normalizeFactor
     
    def _flatten(self, lst):
        result = []
        try:
           for el in lst:
               if hasattr(el, "__iter__") and not isinstance(el, str):
                   result.extend(self._flatten(el))
               else:
                   result.append(el)
        except TypeError:
            result.append(lst)
        return result
        
    def _getStartVectors(self,steps):
       domains = self._getDomains()
       rangeDomains = []
       for i in sorted(domains.keys()):
          if self.index[i] == None:
             start = domains[i][0]
             stop = domains[i][-1]
             step = (stop-start)/steps
             rangeDomains.append(list(np.arange(start, stop+1, step)))
       ranges = []
       for i in range(len(rangeDomains)):
          if i == 0:
             ranges = rangeDomains[i]
          else:
             ranges = [list([j,k]) for j in ranges for k in rangeDomains[i]]
       ranges = [self._flatten(i) for i in ranges]
       return ranges
           


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()

    data = pd.DataFrame(data=np.c_[iris['data']],
                        columns=iris['feature_names'])

    mspn = MSPNModel("Iris")
    mspn.set_data(data)
    mspn.fit()
    domains = mspn._getDomains()
    rangeDomains = []
    for i in sorted(domains.keys()):
       start = domains[i][0]
       stop = domains[i][-1]
       step = (stop-start)/10
       print(start, stop, step)
       rangeDomains.append(list(np.arange(start, stop+1, step)))
    ranges = []
    for i in range(len(rangeDomains)):
       if i == 0:
          ranges = rangeDomains[i]
       else:
          ranges = [list([j,k]) for j in ranges for k in rangeDomains[i]]
    ranges = [mspn._flatten(i) for i in ranges]
    
       