#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13:49:15 2019

@author: Christian Lengert
@email: christian.lengert@dlr.de
"""

import copy as cp
import functools
import os
from pathlib import Path

import numpy as np
from numpy.random.mtrand import RandomState
import scipy.optimize as scpo
import pandas as pd
import dill

from spn.algorithms.stats.Expectations import Expectation
from spn.structure.Base import Context
from spn.algorithms.LearningWrappers import learn_parametric, learn_mspn
from spn.algorithms.Marginalization import marginalize
# from spn.algorithms.Inference import eval_spn_bottom_up
from spn.algorithms.Inference import likelihood
from spn.algorithms.Sampling import sample_instances
from spn.io.Graphics import plot_spn, plot_spn_to_svg

from mb.modelbase import core as core
from mb.modelbase import utils


class SPFlowModel(core.Model):
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
                 spn_type=None):
        super().__init__(name)
        self._spn_type = spn_type
        self._aggrMethods = {
            'maximum': self._maximum,  # TODO use maximum
            'expectation': self._expectation
        }
        self._unbound_updater = functools.partial(self.__class__._update, self)

    def _update(self):
        """
        the _state_mask is updated when the model is marginalized or conditioned
        marginalized variables are represented with np.nan according to spflow
        conditioned variables are represented with 2
        untouched variables are represented with 1
        """
        self._state_mask = np.array(
            [
                np.nan if i in self._marginalized else 1
                if i in self._conditioned else 2
                for i in self._initial_names
            ]
        ).reshape(-1, self._initial_names_count).astype(float)
        return self

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)

        data = self.data
        all, cat, num = utils.get_columns_by_dtype(data)

        # Construct pandas categorical for each categorical variable

        self._categorical_variables = {
            name: {'categorical': pd.Categorical(data[name])} for name in cat
        }

        # Construct inverse dictionary for all categorical variables to use in the function _density
        for k, v in self._categorical_variables.items():
            name_to_int = dict()
            int_to_name = dict()
            categorical = v['categorical']
            expressions = categorical.unique()

            for i, name in enumerate(expressions):
                name_to_int[name] = i
                int_to_name[i] = name

            v['name_to_int'] = name_to_int
            v['int_to_name'] = int_to_name

    def set_var_types(self, var_types):
        self.var_types = var_types

    def set_spn_type(self, spn_type):
        self._spn_type = spn_type

    def _fit(self, var_types=None, **kwargs):
        if self._spn_type == None:
            raise Exception("No SPN-type provided")

        if var_types != None:
            self.var_types = var_types
        else:
            var_types = self.var_types

        df = self.data.copy()
        # Exchange all object columns for their codes as SPFLOW cannot deal with Strings
        for key, value in self._categorical_variables.items():
            df[key] = value['categorical'].codes

        self._nameToVarType = var_types

        # Check if variable types are given
        if self._nameToVarType is None:
            raise ValueError("missing argument 'var_types'")

        self._initial_names = self.names.copy()
        self._initial_names_count = len(self._initial_names)
        self._initial_names_to_index = {self._initial_names[i]: i for i in range(self._initial_names_count)}

        # Initialize _state_mask with np.nan
        self._state_mask = np.array(
            [np.nan for i in self._initial_names]
        ).reshape(-1, self._initial_names_count).astype(float)

        # Initialize _condition with np.nan
        self._condition = np.repeat(
            np.nan,
            self._initial_names_count
        ).reshape(-1, self._initial_names_count).astype(float)

        self._marginalized = set()
        self._conditioned = set()

        try:
            var_types = [self._nameToVarType[name] for name in self.names]
        except KeyError as err:
            raise ValueError('missing var type information for dimension: {}.'.format(err.args[0]))

        if self._spn_type == 'spn':
            context = Context(parametric_types=var_types).add_domains(df.values)
            self._spn = learn_parametric(df.values, context)

        elif self._spn_type == 'mspn':
            context = Context(meta_types=var_types).add_domains(df.values)
            self._spn = learn_mspn(df.values, context)
        else:
            raise Exception("Type of SPN not known: " + self._spn_type)

        # if self._spn:
        #     plot_spn(self._spn, fname=Path(f"../../bin/experiments/spn_graphs/{self.name}.pdf"))
        #     plot_spn_to_svg(self._spn, fname=Path(
        #         f"../../bin/experiments/spn_graphs/{self.name}.svg"))
        return self._unbound_updater,

    def _marginalizeout(self, keep, remove):
        self._marginalized = self._marginalized.union(remove)
        self._spn = marginalize(self._spn, [self._initial_names_to_index[x] for x in keep])
        return self._unbound_updater,

    def _conditionout(self, keep, remove):
        self._conditioned = self._conditioned.union(remove)
        condition_values = self._condition_values(remove)

        for i in range(len(remove)):
            if remove[i] in self._categorical_variables:
                self._condition[:, i] = self._categorical_variables[remove[i]]['name_to_int'][condition_values[i]]
            else:
                self._condition[:, i] = condition_values[i]

        return self._unbound_updater,

    def _opt_density(self, x):
        """A dedicated density function that does not convert the input to numerical arguments like
         _density.

        Used in the _maximum as objective function for the optimizer.
        """
        state_mask = self._state_mask.copy()

        # Values of x get written to the respective positions in the state mask
        counter = 0
        for i in range(state_mask.shape[1]):
            # if variable on index i is not conditioned or marginalized
            # set input i to the value in the input array indicated by counter
            if state_mask[0, i] == 2:
                state_mask[0, i] = x[counter]
                counter += 1
            # if the variable i is conditioned set the input value to the condition
            elif state_mask[0, i] == 1:
                state_mask[0, i] = self._condition[0, i]
            # else the value of input at index i is np.nan by initialization and indicates a marginalized variable

        res = likelihood(self._spn, state_mask)
        return res[0][0]

    def _density(self, x):
        # map all inputs from categorical to numeric values
        x = self._names_to_numeric(x)

        # Copy the current state of the network
        input = self._state_mask.copy()
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

    def save(self, dir, filename=None, *args, **kwargs):
        """Store the model to a file at `filename`.

        You can load a stored model using `Model.load()`.
        """
        if filename is None:
            filename = self._default_filename()
        path = os.path.join(dir, filename)

        with open(path, 'wb') as output:
            dill.dump(self, output, dill.HIGHEST_PROTOCOL)

    def _numeric_to_names(self, x):
        """Convert the categorical variables in the input vector from their numerical representation
         to the a string representation.
        """
        for i in range(len(x)):
            if self.names[i] in self._categorical_variables:
                x[i] = self._categorical_variables[self.names[i]]['int_to_name'][round(x[i])]
        return x

    def _names_to_numeric(self, x: list):
        """Convert the categorical variables in the input list.
        """
        for i in range(len(x)):
            if self.names[i] in self._categorical_variables:
                x[i] = self._categorical_variables[self.names[i]]['name_to_int'][x[i]]
        return x

    def _expectation(self):
        e = Expectation(self._spn)[0].tolist()
        res = self._numeric_to_names(e)
        return res

    def _maximum(self) -> list:
        fun = lambda x: -1 * self._opt_density(x)
        n_samples = 10  # TOOD: make as argument!
        samples = self.data.sample(n_samples)
        numeric_samples = [self._names_to_numeric(samples.iloc[i, :].tolist())
                           for i in range(samples.shape[0])]

        optima = [ scpo.minimize(fun, np.array(x), method='Nelder-Mead') for x in numeric_samples ]
        maxima = [ x['x'] for x in optima]
        # NIPS2020: returns here
        return max(maxima).tolist()

        # CL version continues as follows:
        #values = [ x['fun'] for x in optima ]

        #x0 = np.random.rand(len(self.data.columns.values))
        #xopt = scpo.minimize(fun, x0, method='Nelder-Mead')
        #res = self._numeric_to_names(list(xopt['x']))
        #return res

    def _sample(self, n=1, random_state=RandomState(123)):
        placeholder = np.repeat(np.array(self._condition), n, axis=0)
        s = sample_instances(self._spn, placeholder, random_state)
        indices = [self._initial_names_to_index[name] for name in self.names]
        result = s[:, indices]
        result = [self._numeric_to_names(l) for l in result.tolist()]
        return result

        # master (CL) instead does:
        # result = result.tolist()
        # names = self.names
        # # convert integers back to categorical names
        # for r in result:
        #    for i in range(len(r)):
        #        if names[i] in self._categorical_variables:
        #            r[i] = self._categorical_variables[names[i]]['int_to_name'][round(r[i])]
        # return result

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._spn = cp.deepcopy(self._spn)
        mycopy._spn_type = cp.copy(self._spn_type)
        mycopy._nameToVarType = cp.copy(self._nameToVarType)
        mycopy._initial_names = self._initial_names.copy()
        mycopy._initial_names_count = len(mycopy._initial_names)
        mycopy._marginalized = self._marginalized.copy()
        mycopy._conditioned = self._conditioned.copy()
        mycopy._state_mask = self._state_mask.copy()
        mycopy._condition = self._condition.copy()
        mycopy._categorical_variables = self._categorical_variables.copy()
        mycopy._initial_names_to_index = self._initial_names_to_index.copy()
        return mycopy


if __name__ == "__main__":
    # from sklearn.datasets import load_iris
    # iris_data = load_iris()

    iris_data = pd.read_csv('/home/leng_ch/git/lumen/datasets/mb_data/iris/iris.csv')
    print(iris_data)
    spn = SPFlowModel(name="spn_test", spn_type='spn')
    import spn.structure.leaves.parametric.Parametric as spn_parameter_types

    var_types = {
        'sepal_length': spn_parameter_types.Gaussian,
        'sepal_width': spn_parameter_types.Gaussian,
        'petal_length': spn_parameter_types.Gaussian,
        'petal_width': spn_parameter_types.Gaussian,
        'species': spn_parameter_types.Categorical}
    spn.fit(df=pd.DataFrame(iris_data), var_types=var_types)

    spn.save('/home/leng_ch/git/lumen/fitted_models')

    with open("/home/leng_ch/git/lumen/fitted_models/spn_test.mdl", "rb") as f:
        test_model = dill.load(f)
