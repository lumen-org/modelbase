# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de), Jonas Gütter (jonas.aaron.guetter@uni-jena.de)
import os
from mb_modelbase.models_core.models import Model
from mb_modelbase.utils.data_import_utils import get_numerical_fields
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
import pymc3 as pm
import numpy as np
import pandas as pd
import math
from mb_modelbase.models_core.empirical_model import EmpiricalModel
from mb_modelbase.models_core import data_operations as data_op
from scipy import stats
import scipy.optimize as sciopt
import copy as cp
from functools import reduce

class ProbabilisticPymc3Model(Model):
    """
    A Bayesian model built by the PyMC3 library is treated here.

        Parameters:

        model_structure : a PyMC3 Model() instance

        shared_vars : dictionary of theano shared variables

            If the model has independent variables, they have to be encoded as theano shared variables and provided
            in this dictionary, additional to the general dataframe containing all observed data. Watch out: It is
            NOT guaranteed that shared_vars always holds the original independent variables, since they are changed
            during the _fit()-method

        nr_of_posterior_samples: integer scalar specifying the number of posterior samples to be generated
    """

    def __init__(self, name, model_structure, shared_vars=None, nr_of_posterior_samples=500, nr_of_partitions=1):
        super().__init__(name)
        self.model_structure = model_structure
        self.samples = pd.DataFrame()
        self._aggrMethods = {
            'maximum': self._maximum
        }
        self.parallel_processing = False
        self.shared_vars = shared_vars
        self.nr_of_posterior_samples = nr_of_posterior_samples
        self.nr_of_partitions = nr_of_partitions

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently, split_data=False)
        self._update_all_field_derivatives()
        # Enforce usage of theano shared variables for independent variables
        # Independent variables are those variables which appear in the data but not in the RVs of the model structure
        model_vars = [str(name) for name in self.model_structure.observed_RVs]
        ind_vars = [varname for varname in self.data.columns.values if varname not in model_vars]
        # When there are no shared variables, there should be no independent variables. Otherwise, raise an error
        if not self.shared_vars:
            assert len(ind_vars) == 0, \
                'The model appears to include the following independent variables: ' + str(ind_vars) + ' It is '\
                'required to pass the data for these variables as theano shared variables to the ' \
                'ProbabilisticPymc3Model constructor'
        # When there are shared variables, there should be independent variables. Otherwise, raise an error
        else:
            assert len(ind_vars) > 0, ' theano shared variables were passed to the ProbabilisticPymc3Model constructor'\
                                      ' but the model does not appear to include independent variables. Only pass '\
                                      'shared variables to the constructor if the according variables are independent'
            # Each independent variable should appear in self.shared_vars. If not, raise an error
            missing_vars = [varname for varname in ind_vars if varname not in self.shared_vars.keys()]
            assert len(missing_vars) == 0, \
                'The following independent variables do not appear in shared_vars:' + str(missing_vars) + ' Make sure '\
                'that you pass the data for each independent variable as theano shared variable to the constructor'
        return ()

    def _check_for_datadependent_priors(self):
        if self.shared_vars:
            data_dependent_prior = False
            for varname in self.model_structure.unobserved_RVs:
                # Get a probability from the prior
                logp1 = varname.distribution.logp(0)
                # Change independent data
                old_ind = cp.deepcopy(self.shared_vars)
                for key,value in self.shared_vars:
                    new_ind = np.random.uniform(0,1,size=len(value))
                    self.shared_vars[key].set_value(new_ind)
                # Get a second probability from the prior
                logp2 = varname.distribution.logp(0)
                # The two probabilities should be equal, if the prior is not dependent on the data
                if logp1 != logp2:
                    data_dependent_prior = True
                # Change independent variables back to previous values
                for key,value in self.shared_vars:
                    self.shared_vars[key].set_value(old_ind[key])

            if data_dependent_prior:
                raise ValueError('A parameter of the model seems to be directly parametrized by data. '
                                 'This kind of model is not supported')

    def _generate_samples_for_independent_variable(self, key, size):
        lower_bound = self.byname(key)['extent'].value()[0]
        upper_bound = self.byname(key)['extent'].value()[1]
        generated_samples = np.linspace(lower_bound, upper_bound, num=size)
        # If the samples have another data type than the original data, problems can arise. Therefore,
        # data types of the new samples are changed to the dtypes of the original data here
        if str(generated_samples.dtype) != self.shared_vars[key].dtype:
            generated_samples = generated_samples.astype(self.shared_vars['years'].dtype)
        return generated_samples

    def _cartesian_product_of_samples(self, df):
        # Split up df in its columns
        dfs = [pd.DataFrame(df[col]) for col in df]
        # Add equal column to each split
        [df.insert(1, 'key', 1) for df in dfs]
        # cross join all dfs
        cartesian_prod = reduce(lambda left, right: pd.merge(left, right, on='key'), dfs)
        cartesian_prod = cartesian_prod.drop('key', 1)
        return cartesian_prod

    def _fit(self):
        # Generate samples for latent random variables
        with self.model_structure:
            for var in self.fields:
                self.samples[var['name']] = np.full(self.nr_of_posterior_samples, np.NaN)
            trace = pm.sample(self.nr_of_posterior_samples, chains=1, cores=1, progressbar=False)
            # Store varnames for later generation of fields
            varnames = trace.varnames.copy()
            for varname in trace.varnames:
                # check if trace consists of more than one variable
                if len(trace[varname].shape) == 2:
                    varnames.remove(varname)
                    for i in range(trace[varname].shape[1]):
                        self.samples[varname+'_'+str(i)] = [var[i] for var in trace[varname]]
                        varnames.append(varname+'_'+str(i))
                else:
                    self.samples[varname] = trace[varname]
        # Generate samples for observed independent variables
        with self.model_structure:
            if self.shared_vars is not None:
                # Values at equal intervals over the domain of each independent variable are generated. Then a grid of
                # values is constructed so that each combination of values between the different variables is covered
                nr_of_ind_vars = len(self.shared_vars)
                nr_of_unique_values = self.nr_of_posterior_samples ** (1. / nr_of_ind_vars)
                generated_samples = pd.DataFrame(columns=self.shared_vars.keys())
                for key, val in self.shared_vars.items():
                    generated_samples[key] = self._generate_samples_for_independent_variable(key, nr_of_unique_values)
                samples_independent_vars = self._cartesian_product_of_samples(generated_samples)
                # number of samples in samples_independent_vars may now be lesser than self.nr_of_posterior_samples.
                # The first x rows of samples_independent_vars are therefore attached again at the end of the
                # dataframe, where x is the difference to self.nr_of_posterior_samples
                diff = self.nr_of_posterior_samples - len(samples_independent_vars)
                samples_independent_vars = samples_independent_vars.append(samples_independent_vars.iloc[0:diff, :])
                #
                for col in samples_independent_vars:
                    #self.shared_vars[col].set_value(samples_independent_vars[col])
                    self.samples[col] = samples_independent_vars[col].values
        # Generate samples for observed dependent variables
        with self.model_structure:
            obs_per_partition = self.nr_of_posterior_samples/self.nr_of_partitions
            for i in range(self.nr_of_partitions):
                lower_idx = i * obs_per_partition
                upper_idx = (i + 1) * obs_per_partition
                for col in samples_independent_vars:
                    self.shared_vars[col].set_value(samples_independent_vars[col].iloc[lower_idx:upper_idx, :])
                ppc = pm.sample_ppc(trace)
                if self.shared_vars is not None:
                    # sample_ppc works the following way: For each parameter set generated by sample(), a sequence
                    # of points is generated with the same length as the observed data.
                    for varname in self.model_structure.observed_RVs:
                        self.samples[str(varname)][lower_idx:upper_idx] = \
                            [ppc[str(varname)][j][j] for j in range(obs_per_partition)]
                else:
                    # when no shared vars are given, data and samples do not have the same length. In this case, the first
                    # point of each sequence is taken as new sample point
                    for varname in self.model_structure.observed_RVs:
                        self.samples[str(varname)] = [samples[0] for samples in ppc[str(varname)]]


        # Add parameters to fields
        self.fields = self.fields + get_numerical_fields(self.samples, varnames)
        self._update_all_field_derivatives()
        self._init_history()

        # Change order of sample columns so that it matches order of fields
        self.samples = self.samples[self.names]
        self.test_data = self.samples

        # Mark variables as independent. Independent variables are variables that appear in the data but
        # not in the observed random variables of the model
        for field in self.fields:
            if field['name'] in self.data.columns and \
                    field['name'] not in [str(var) for var in self.model_structure.observed_RVs]:
                field['independent'] = True
        return ()

    def _marginalizeout(self, keep, remove):
        keep_not_in_names = [name for name in keep if name not in self.names]
        if len(keep_not_in_names) > 0:
            raise ValueError('The following variables in keep do not appear in the model: ' + str(keep_not_in_names) )
        remove_not_in_names = [name for name in remove if name not in self.names]
        if len(remove_not_in_names) > 0:
            raise ValueError('The following variables in remove do not appear in the model: ' + str(remove_not_in_names))
        # Remove all variables in remove
        for varname in remove:
            if varname in list(self.samples.columns):
                self.samples = self.samples.drop(varname,axis=1)
            if hasattr(self, 'shared_vars'):
                if self.shared_vars is not None:
                    if varname in self.shared_vars:
                        del self.shared_vars[varname]
        return ()

    def _conditionout(self, keep, remove):
        keep_not_in_names = [name for name in keep if name not in self.names]
        if len(keep_not_in_names) > 0:
            raise ValueError('The following variables in keep do not appear in the model: ' + str(keep_not_in_names) )
        remove_not_in_names = [name for name in remove if name not in self.names]
        if len(remove_not_in_names) > 0:
            raise ValueError('The following variables in remove do not appear in the model: ' + str(remove_not_in_names))
        names = remove
        fields = [] if names is None else self.byname(names)
        # Konditioniere auf die Domäne der Variablen in remove
        for field in fields:
            # filter out values smaller than domain minimum
            filter = self.samples.loc[:, str(field['name'])] > field['domain'].value()[0]
            self.samples.where(filter, inplace=True)
            # filter out values bigger than domain maximum
            filter = self.samples.loc[:, str(field['name'])] < field['domain'].value()[1]
            self.samples.where(filter, inplace=True)
        self.samples.dropna(inplace=True)
        self._marginalizeout(keep, remove)
        return ()

    def _density(self, x):
        if any([self.fields[i]['independent'] for i in range(len(self.fields))]):
            #raise ValueError("Density is queried for a model with independent variables")
            return np.NaN
        elif self.samples.empty | len(self.samples) == 1:
            #raise ValueError("There are not enough samples in the model")
            return np.NaN
        else:
            X = self.samples.values
            kde = stats.gaussian_kde(X.T)
            x = np.reshape(x, (1, len(x)))
            density = kde.evaluate(x)[0]
            return density

    def _negdensity(self,x):
        return -self._density(x)

    def _sample(self):
        sample = []
        with self.model_structure:
            trace = pm.sample(1,chains=1,cores=1)
            ppc = pm.sample_ppc(trace)
            for varname in self.names:
                if varname in [str(name) for name in self.model_structure.free_RVs]:
                    sample.append(trace[varname][0])
                elif varname in [str(name) for name in self.model_structure.observed_RVs]:
                    sample.append(ppc[str(varname)][0][0])
                else:
                    raise ValueError("Unexpected error: variable name " + varname +  " is not found in the PyMC3 model")

        return (sample)

    def copy(self, name=None):
        name = self.name if name is None else name
        mycopy = self.__class__(name, self.model_structure)
        mycopy.data = self.data.copy()
        mycopy.test_data = self.test_data.copy()
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy.mode = self.mode
        mycopy._update_all_field_derivatives()
        mycopy.history = cp.deepcopy(self.history)
        mycopy.samples = self.samples.copy()
        return mycopy

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        row_cnt, col_cnt = self.samples.shape
        if row_cnt == 0:
            # can not compute any aggregation. return nan
            return [None] * col_cnt
        # Set starting point for optimization problem
        x0 = [np.mean(self.samples[col]) for col in self.samples]
        opt = sciopt.minimize(self._negdensity, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
        maximum = opt.x
        # Do not return a value if the density of the found maximum is NaN. In this case it is assumed that all
        # density values are NaN, so there should not be returned a maximum
        if np.isnan(opt.fun):
            return np.full(len(x0), np.nan)
        return maximum


