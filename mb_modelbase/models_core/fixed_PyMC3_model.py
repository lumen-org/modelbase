# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de), Jonas Gütter (jonas.aaron.guetter@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.utils.data_import_utils import get_numerical_fields
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
import pymc3 as pm
import numpy as np
import pandas as pd
from mb_modelbase.models_core.empirical_model import EmpiricalModel
from mb_modelbase.models_core import data_operations as data_op
from sklearn.neighbors.kde import KernelDensity

#Specify the example model here
data = pd.read_csv('/home/philipp/Desktop/code/mb_data/mb_data/jonas_guetter/fixed_PyMC3_example_data.csv')

basic_model = pm.Model()
with basic_model:
    # describe prior distributions of model parameters.
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=1, sd=5)
    beta2 = pm.Normal('beta2', mu=2, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)
    X1 = pm.Normal('X1',mu=0, sd=1, observed=data['X1'])
    X2 = pm.Normal('X2', mu=0, sd=0.2, observed=data['X2'])
    # specify model for the output parameter.
    mu = alpha + beta1 * X1 + beta2 * X2
    # likelihood of the observations. Observed stochastic variable
    Y = pm.Normal('Y', mu=mu, sd=sigma, observed= data['Y'])


class FixedProbabilisticModel(Model):
    """
    A Bayesian model built by the PyMC3 library is treated here.
    """

    def __init__(self, name):
        super().__init__(name)


    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        self._update_all_field_derivatives()
        return ()

    def _fit(self):
        with basic_model:
            # Draw samples
            colnames = [str(name) for name in basic_model.observed_RVs] + [str(name) for name in basic_model.unobserved_RVs]
            self.samples = pd.DataFrame(columns=colnames)
            nr_of_samples = 500
            trace = pm.sample(nr_of_samples)
            for varname in trace.varnames:
                self.samples[varname] = trace[varname]
            # samples above were drawn for 4 chains by default.
            # ppc samples are drawn 100 times for each sample by default
            ppc = pm.sample_ppc(trace, samples=int(nr_of_samples*4/100), model=basic_model)
            for varname in basic_model.observed_RVs:
                self.samples[str(varname)] = np.asarray(ppc[str(varname)].flatten())
            self.test_data = self.samples
            # Add parameters to fields
            self.fields = self.fields + get_numerical_fields(self.samples, trace.varnames)
            self._update_all_field_derivatives()
            self._init_history()
        return ()

# Achtung: _marginalizeout is currently only for parameters possible,
# whereas _conditionout is only for data points possible

    def _marginalizeout(self, keep, remove):
        # Remove all variables in remove
        for varname in remove:
            if varname in list(self.samples.columns):
                self.samples = self.samples.drop(varname,axis=1)
        return ()

    def _conditionout(self, keep, remove):
        names = remove
        fields = self.fields if names is None else self.byname(names)
        cond_domains = [field['domain'] for field in fields]
        # Here: Konditioniere auf die Domäne der Variablen in remove
        for field in fields:
            # filter out values smaller than domain minimum
            filter = self.samples.loc[:,str(field['name'])] > field['domain'].value()[0]
            self.samples.loc[:,str(field['name'])].where(filter, inplace = True)
            # filter out values bigger than domain maximum
            filter = self.samples.loc[:,str(field['name'])] < field['domain'].value()[1]
            self.samples.loc[:,str(field['name'])].where(filter, inplace = True)
        return ()

    def _density(self, x):
        X = self.samples.values
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
        x = np.reshape(x,(1,len(x)))
        logdensity = kde.score_samples(x)
        return (np.exp(logdensity))

    def _sample(self):
        with basic_model:
            trace = pm.sample(1)
            point = trace.point(0)
        return (point)

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy.samples = self.samples
        return (mycopy)
