# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)
# Copyright (c) 2019 Jonas Gütter (jonas.aaron.guetter@uni-jena.de)

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as sciopt
import copy as cp
from functools import reduce
import theano
import math

from mb_modelbase.models_core.models import Model
from mb_modelbase.utils.data_import_utils import get_numerical_fields


class ProbabilisticPymc3Model(Model):
    """A Bayesian model built by the PyMC3 library is treated here.

    Parameters:

        model_structure : a PyMC3 Model() instance

        shared_vars : dictionary of theano shared variables

            If the model has independent variables, they have to be encoded as theano shared variables and provided
            in this dictionary, additional to the general dataframe containing all observed data. Watch out: It is
            NOT guaranteed that shared_vars always holds the original independent variables, since they are changed
            during the _fit()-method

        nr_of_posterior_samples: integer scalar specifying the number of posterior samples to be generated

        fixed_data_length: boolean, indicates if the model requires the data to have a fixed length

            Some probabilistic models require a fixed length of the data. This is important because normally
            new data points are generated with a different length than the original data
    """

    def __init__(self, name, model_structure, shared_vars=None, nr_of_posterior_samples=500, fixed_data_length=False):
        super().__init__(name)
        self.model_structure = model_structure
        self.samples = pd.DataFrame()
        self._aggrMethods = {
            'maximum': self._maximum
        }
        self.parallel_processing = False
        self.shared_vars = shared_vars
        self.nr_of_posterior_samples = nr_of_posterior_samples
        self.fixed_data_length = fixed_data_length

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
            generated_samples = generated_samples.astype(self.shared_vars[key].dtype)
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
        self.samples = self._sample(self.nr_of_posterior_samples)

        # Add parameters to fields
        latent_fields = get_numerical_fields(self.samples, [str(var) for var in self.model_structure.unobserved_RVs])
        for f in latent_fields:
            f['obstype'] = 'latent'

        self.fields = self.fields + latent_fields
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

    def _negdensity(self, x):
        return -self._density(x)

    def _sample(self, n):
        sample = pd.DataFrame()

        # Generate samples for latent random variables
        with self.model_structure:
            trace = pm.sample(n, chains=1, cores=1, progressbar=False)
            for varname in trace.varnames:
                # check if trace consists of more than one variable
                if len(trace[varname].shape) == 2:
                    for i in range(trace[varname].shape[1]):
                        sample[varname+'_'+str(i)] = [var[i] for var in trace[varname]]
                else:
                    sample[varname] = trace[varname]

        # Generate samples for observed independent variables
        if self.shared_vars is not None:
            samples_independent_vars = pd.DataFrame(columns=self.shared_vars.keys())
            data_independent_vars = self.data.loc[:, self.shared_vars.keys()]
            # Draw without replacement from the observed data. If more values should be drawn than there are in the
            # data, take the whole data multiple times
            data_fits_in_n = math.floor(n/len(self.data))
            for i in range(data_fits_in_n):
                samples_independent_vars = samples_independent_vars.append(data_independent_vars.copy())
            samples_independent_vars = samples_independent_vars.append(
                data_independent_vars.sample(n-data_fits_in_n*len(self.data), replace=False))
            for varname in samples_independent_vars:
                sample[varname] = samples_independent_vars[varname].values

        # Generate samples for observed dependent variables
        if self.shared_vars is not None:
            for col in samples_independent_vars:
                self.shared_vars[col].set_value(samples_independent_vars[col])
        with self.model_structure:
            ppc = pm.sample_ppc(trace, size=self.nr_of_posterior_samples)
            # sample_ppc works the following way: For each parameter set generated by pm.sample(), a sequence
            # of points is generated with the length specified in the size argument.
            for varname in self.model_structure.observed_RVs:
                sample[str(varname)] = [ppc[str(varname)][j][j] for j in range(ppc[str(varname)].shape[0])]

        return sample

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

        #Copy shared_vars
        mycopy.shared_vars = {}
        if self.shared_vars:
            for key, value in self.shared_vars.items():
                mycopy.shared_vars[key] = theano.shared(value.get_value().copy())
        mycopy.nr_of_posterior_samples = self.nr_of_posterior_samples
        mycopy.fixed_data_length = self.fixed_data_length
        mycopy.set_empirical_model_name(self._empirical_model_name)

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

