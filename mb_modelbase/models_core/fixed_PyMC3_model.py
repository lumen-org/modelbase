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
import copy as cp

class FixedProbabilisticModel(Model):
    """
    A Bayesian model built by the PyMC3 library is treated here.
    """

    def __init__(self, name, model_structure):
        super().__init__(name)
        self.model_structure = model_structure


    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        self._update_all_field_derivatives()
        return ()

    def _fit(self):
        with self.model_structure:
            # Draw samples
            colnames = ['mu','X']
            self.samples = pd.DataFrame(columns=colnames)
            nr_of_samples = 500
            trace = pm.sample(nr_of_samples)
            for varname in trace.varnames:
                self.samples[varname] = trace[varname]
            # samples above were drawn for 4 chains by default.
            # ppc samples are drawn 100 times for each sample by default
            #ppc = pm.sample_ppc(trace, samples=int(nr_of_samples*4/100), model=self.model_structure)
            #for varname in self.model_structure.observed_RVs:
            #    self.samples[str(varname)] = np.asarray(ppc[str(varname)].flatten())
            size_ppc = len(trace['mu'])
            self.samples['X'] = np.random.normal(self.samples['mu'],1,size=size_ppc)
            self.test_data = self.samples
            # Add parameters to fields
            self.fields = self.fields + get_numerical_fields(self.samples, trace.varnames)
            self._update_all_field_derivatives()
            self._init_history()
        return ()

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

    # First column of self.samples.values is mu, second column is x
    # Currently only works for a single point
    def _density(self, x):
        X = self.samples.values
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
        x = np.reshape(x,(1,len(x)))
        logdensity = kde.score_samples(x)
        return (np.exp(logdensity))

    def _sample(self):
        with self.model_structure:
            trace = pm.sample(1)
            point = trace.point(0)
        return (point)

    def copy(self, name=None):
        name = self.name if name is None else name
        mycopy = self.__class__(name, self.model_structure)
        mycopy.data = self.data  # .copy()
        mycopy.test_data = self.test_data.copy()
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy.mode = self.mode
        mycopy._update_all_field_derivatives()
        mycopy.history = cp.deepcopy(self.history)
        mycopy.samples = self.samples
        #mycopy.model_structure = self.model_structure
        return (mycopy)

if __name__ == '__main__':
    from mb_modelbase.models_core.fixed_PyMC3_model import *
    from mb_modelbase.models_core import auto_extent
    import matplotlib.pyplot as plt

    # Generate data
    np.random.seed(2)
    size = 100
    mu = np.random.normal(0, 1, size=size)
    sigma = 1
    X = np.random.normal(mu, sigma, size=size)

    data = pd.DataFrame({'X': X})

    # Create model
    basic_model = pm.Model()
    with basic_model:
        mu = pm.Normal('mu', mu=0, sd=1)
        X = pm.Normal('X', mu=mu, sd=1, observed=X)

        nr_of_samples = 10000
        trace = pm.sample(nr_of_samples, tune=1000, cores=4)

    modelname = 'my_pymc3_model'
    m = FixedProbabilisticModel(modelname,basic_model)
    m.fit(data)
    Model.save(m, '../../../mb_data/data_models/{}.mdl'.format(modelname))

    # traceplot of mu from basic example to generate traceplot_basic_pymc3_example_tocomparemuwithgroundtruth.png

    # Generate data
    # np.random.seed(2)
    # size = 100
    # mu = np.random.normal(0, 1, size=size)
    # sigma = 1
    # X = np.random.normal(mu, sigma, size=size)
    #
    # data = pd.DataFrame({'X': X})
    #
    # # Create model
    # basic_model = pm.Model()
    # with basic_model:
    #     mu = pm.Normal('mu', mu=0, sd=1)
    #     X = pm.Normal('X', mu=mu, sd=1, observed=X)
    #
    #     nr_of_samples = 10000
    #     trace = pm.sample(nr_of_samples, tune=10000, cores=4)
    #
    # pm.traceplot(trace)
    # plt.show()

