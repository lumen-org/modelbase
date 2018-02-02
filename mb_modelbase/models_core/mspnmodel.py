#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:48:00 2018

@author: julien
@email: julien.klaus@uni-jena.de

Note to get it run:

In addition to any dependencies required by the mb_modelbase package you also need:
    (1) these python packages available:
        rpy2, joblib, tensorflow
      to install pip install rpy2, joblib, tensorflow
    (2) R installed, with these packages available:
        histogram, iterators, partykit, foreach, doMC, dplyr, gtools, igraph, energy, dummies,
      to install, run from an R console:
        install.package(histogram, iterators, partykit, foreach, doMC, dplyr, gtools, igraph, energy, dummies)


What did we do to make it run:
  * in particular we changed line ~ tfspn.py:1230
"""
import rpy2

from mb_modelbase.models_core import Model, get_columns_by_dtype

from mb_modelbase.models_core.mspn.tfspn.piecewise import estimate_domains
from mb_modelbase.models_core.mspn.tfspn.SPN import SPN, Splitting

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import copy as cp


class MSPNModel(Model):
    def __init__(self, name, threshold=None, min_instances_slice=None):
        super().__init__(name)
        self.featureTypes = None
        self.threshold = threshold if threshold is not None else 0.4
        self.min_instances_slice = min_instances_slice
        self.index = {}
        self._aggrMethods = {
            'maximum': self._maximum,
        }

    def _set_data(self, df, drop_silently=False):
        _, categoricals, numericals = get_columns_by_dtype(df)
        self._set_data_mixed(df, drop_silently)
        self.featureTypes = ["categorical"]*len(categoricals) + ['continuous']*len(numericals)
        if self.min_instances_slice is None:
            # if there is not minimum slice width we set it to 8% of the data length
            self.min_instances_slice = len(self.data) * 0.08
        self.nametoindex = dict((field["name"], i) for i, field in zip(range(len(self.fields)), self.fields))
        for i in range(len(self.fields)):
            self.index[i] = None
        return []

    def _fit(self):
        self._mspnmodel = SPN.LearnStructure(np.array(self.data), featureTypes=self.featureTypes, \
                                             row_split_method=Splitting.KmeansRows(), \
                                             col_split_method=Splitting.RDCTest(threshold=self.threshold), \
                                             min_instances_slice=self.min_instances_slice).root
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
        return np.exp(self._mspnmodel.eval(None, tmp))

    # calculated iterations times the maximum and returns the position
    # with the highest densitiy value
    def _maximum(self, iterations=10):
        fun = lambda x: -1 * self._density(x)
        xlength = sum(1 for x in self.index.values() if x is None)
        xmax = None
        for i in range(iterations):
            x0 = np.random.randn(xlength)
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


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()

    data = pd.DataFrame(data=np.c_[iris['data']],
                        columns=iris['feature_names'])

    mspn = MSPNModel("Iris")
    mspn.set_data(data)
    mspn.fit()
    mspn.marginalize(remove=[mspn.names[0]])
    data2 = np.array([4.4, 4.4, 2.3])  # , 1.0])
    print(mspn._density(data2))

    # c.save_pdf_graph("asdjh2.pdf")
