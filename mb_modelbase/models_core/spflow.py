#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13:49:15 2019

@author: Christian Lengert
@email: christian.lengert@dlr.de
"""

from mb_modelbase.models_core import Model
from spn.structure.Base import Context
from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn
from spn.algorithms.Marginalization import marginalize
from spn.algorithms.Condition import condition
from spn.algorithms.Inference import eval_spn_bottom_up
from spn.algorithms.Inference import likelihood
from spn.algorithms.Sampling import sample_instances
from numpy.random.mtrand import RandomState
from mb_modelbase.utils import data_import_utils as diu
from spn.io.Graphics import plot_spn
import numpy as np
import functools
import pandas as pd
import copy as cp

class SPNModel(Model):
    """
    Parameters:
        .spn_type : string
            - is either 'spn' or 'mspn'
            - indicates whether the model is a regular spn or a mixed spn
        ._nameToVarType :    Dict: string -> spn.structure.StatisticalTypes.MetaType
                        Dict  string -> spn.structure.leaves.parametric.Parametric.{Categorical, Gaussian,...}
            - defines a mapping from the fields to types of variables
            - in case of a mixed SPN a meta-type, e.g. REAL or DISCRETE has to be provided
            - else the types of distributions have to be given, e.g. Categorical or Gaussian

    """
    def __init__(self, name,
                 spn_type = 'spn'):
        super().__init__(name)
        self._spn_type = spn_type
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        self._unbound_updater = functools.partial(self.__class__._update, self)

    def _update(self):
        """
        the _density_mask is updated when the model is marginalized or conditioned
        marginalized variables are represented with np.nan according to spflow
        conditioned variables are represented with 1
        untouched variables are represented with 2
        """
        self._density_mask = np.array(
            [np.nan if i in self._marginalized else 1 if i in self._conditioned else 2 for i in self._initial_names]
        ).reshape(-1, self._initial_names_count).astype(float)
        return self

    def _set_data(self, df, drop_silently, **kwargs):
        all, cat, num = diu.get_columns_by_dtype(df)

        # Construct pandas categorical for each categorical variable
        self._categorical_variables = {
            name: { 'categorical':pd.Categorical(df[name])} for name in cat
        }

        # Construct inverse dictionary for all categorical variables to use in the function _density
        for k,v in self._categorical_variables.items():
            inverse_mapping = dict()
            categorical = v['categorical']
            codes = categorical.codes
            for i, code in enumerate(codes):
                inv = categorical[i]
                inverse_mapping[inv] = code
            v['inverse_mapping'] = inverse_mapping

        self._set_data_mixed(df, drop_silently)

    def _categorical_to_numeric(self, x):
        """bla"""
        for i in range(len(x)):
            if self.names[i] in self._categorical_columns:
                discrete = 5
                x[i] = round(x[i])
        return x

    def _fit(self, var_types = None):

        df = self.data.copy()
        # Exchange all object columns for their codes
        for key, value in self._categorical_variables.items():
            df[key] = value['categorical'].codes

        self._nameToVarType = var_types
        #Check if variable types are given
        assert self._nameToVarType != None
        #Check if enough
        assert len(self._nameToVarType) == len(self.fields)

        self._initial_names = self.names.copy()
        self._initial_names_count = len(self._initial_names)
        self._initial_names_to_index = {self._initial_names[i]: i for i in range(self._initial_names_count)}

        # Initialize _density_mask with np.nan
        self._density_mask = np.array(
            [np.nan for i in self._initial_names]
        ).reshape(-1, self._initial_names_count).astype(float)

        # Initialize _condition with np.nan
        self._condition = np.repeat(
            np.nan,
            self._initial_names_count
        ).reshape(-1, self._initial_names_count).astype(float)

        self._marginalized = set()
        self._conditioned = set()

        var_types = [self._nameToVarType[name] for name in self.names]

        if self._spn_type == 'spn':
            context = Context(parametric_types=var_types).add_domains(df.values)
            self._spn = learn_parametric(df.values, context)

        elif self._spn_type == 'mspn':
            context = Context(meta_types=var_types).add_domains(df.values)
            self._spn = learn_mspn(df.values, context)
        else:
            raise Exception("Type of SPN not known: " + self._spn_type)
        return self._unbound_updater,

    def _marginalizeout(self, keep, remove):
        self._marginalized = self._marginalized.union(remove)
        #self._spn = marginalize(self._spn, self.asindex(keep))
        return self._unbound_updater,

    def _conditionout(self, keep, remove):
        self._conditioned = self._conditioned.union(remove)
        condvalues = self._categorical_to_numeric(self._condition_values(remove))
        old_indices = [self._initial_names_to_index[name] for name in remove]
        for i in range(len(remove)):
            self._condition[0, old_indices[i]] = condvalues[i]
        return self._unbound_updater,

    def _density(self, x):
        # map all inputs from categorical to numeric values
        #x = self._categorical_to_numeric(x)

        for i in range(len(x)):
            if self.names[i] in self._categorical_variables:
                inverse_mapping = self._categorical_variables[self.names[i]]['inverse_mapping']
                x[i] = inverse_mapping[x[i]]
                #x[i] = round(x[i])

        # Copy the current state of the network
        input = self._density_mask.copy()
        counter = 0
        for i in range(input.shape[1]):
            # if variable on index i is not conditioned or marginalized
            # set input i to the value in the input array indicated by counter
            if input[0, i] == 2:
                input[0, i] = x[counter]
                counter += 1
            # if the variable i is conditioned set the input value to the condition
            elif input[0, i] == 1:
                input[0, i] = self._condition[0, i]
            # else the value of input at index i is np.nan by initialization and indicates a marginalized variable

        res = likelihood(self._spn, input)
        return res[0][0]

    def _maximum(self):
        return 3

    def _sample(self, random_state=RandomState(123)):
        placeholder = self._condition.copy()
        s = sample_instances(self._spn, placeholder, random_state)
        indices = [self._initial_names_to_index[name] for name in self.names]
        result = s[:, indices]
        result = result.reshape(len(self.names)).tolist()

        for i in range(len(result)):
            if self.names[i] in self._categorical_variables:
                result[i] = self._categorical_variables[self.names[i]]['categorical'][round(result[i])]
        return result

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._spn = cp.deepcopy(self._spn)
        mycopy._spn_type = cp.copy(self._spn_type)
        mycopy._nameToVarType = cp.copy(self._nameToVarType)
        mycopy._initial_names = self._initial_names.copy()
        mycopy._initial_names_count = len(mycopy._initial_names)
        mycopy._marginalized = self._marginalized.copy()
        mycopy._conditioned = self._conditioned.copy()
        mycopy._density_mask = self._density_mask.copy()
        mycopy._condition = self._condition.copy()
        mycopy._categorical_variables = self._categorical_variables.copy()
        #mycopy._pandas_data_types = self._pandas_data_types.copy()
        mycopy._initial_names_to_index = self._initial_names_to_index.copy()
        return mycopy

    def plot(self, filename):
        plot_spn(self._spn,filename)


