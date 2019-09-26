#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:49:15 2018

@author: julien
@email: julien.klaus@uni-jena.de

What did we do to make it run:
  * we pushed our index through the eval. if the index of the
    RV is True then we return simply 1.0 for the marginalization

IMPORTANT:
  * the model only allows continous data
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

    def _set_data(self, df, drop_silently, **kwargs):
      self._set_data_continuous(df, drop_silently)
      self.variables = len(self.fields)
      # creates a dict of name : index
      # looks strange but is correct
      self.nametoindex = dict((field["name"], i) for i,field in zip(range(len(self.fields)), self.fields))
      for i in range(self.variables):
         self.index[i] = None
      return []

    def _fit(self, iterations=5, **kwargs):
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
    def _maximum(self):
        fun = lambda x :-1 * self._density(x)
        xmax = None
        xlength = sum([1 for i in self.index.values() if i is None])
        startVectors = self._getStartVectors()
        # if the list is empty, we create 10 values by random
        if len(startVectors) == 0:
           for i in range(10):
              startVectors.append(np.random.rand(xlength))
        for x0 in startVectors:
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
   
    def _getStartVectors(self):
       xlength = sum([1 for i in self.index.values() if i is None])
       worklist = [(self._spnmodel.root.children[0], [None]*(xlength+1))]
       values = []
       i = 0
       #fill the values
       while len(worklist) > 0:
          (node,vector) = worklist.pop(0)
          tmp = vector.copy()
          if node.__class__.__name__ is "NormalLeafNode":
             tmp[node.index+1] = node.mean
             values.append(tmp)
          for child in node.children:
             if node.__class__.__name__ is "SumNode":
                tmpChild = tmp.copy()
                tmpChild[0] = i
                i = i+1
                worklist.append((child,tmpChild))
             else:
                worklist.append((child,tmp))
       #sort for the first key
       valuesSorted = (sorted(values,key=lambda x: x[0]))
       #discard all keys with less then xlenght instances
       valuesFiltered = [i for i in valuesSorted if len([b for b in valuesSorted if b[0] == i[0]]) == xlength]
       #get the names of the indexes
       valuesIndexes = list({i[0] for i in valuesFiltered})
       #create a list of start vectors
       vectors = []
       for i in valuesIndexes:
          #all values for a given index (should be xlength much)
          valuesTemp = [j for j in valuesFiltered if j[0] == i]
          #create a vector for with all not None values at the right index
          xt = [None]*xlength
          for i in range(xlength):
              for j in valuesTemp:
                xt[i] = j[i+1] if j[i+1] is not None else xt[i]      
          vectors.append(xt)
       #remove elements which contains a None
       vectors = [i for i in vectors if None not in i]
       return vectors

if __name__ == "__main__":
    from sklearn.datasets import load_iris


    iris = load_iris()

    data = pd.DataFrame(data=np.c_[iris['data']],
                        columns=iris['feature_names'])

    spn = SPNModel("test")
    spn.set_data(data)
    spn._fit()
    x = spn._maximum()
    print(x, spn._density(x))
    print("start aggregating")
    
    