# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
from sklearn.neighbors.kde import KernelDensity
import copy
import numpy as np


class KDEModel(Model):
    """
    A Kernel Density Estimator (KDE) model is a model whose distribution is determined by using a kernel
    density estimator. KDE work in a way that to each point of the observed data a distribution is
    assigned centered at that point (e.g. a normal distribution). The distributions from all data points
    are then summed up and build up the joint distribution for the model
    (see https://scikit-learn.org/stable/modules/density.html#kernel-density)
    """

    def __init__(self, name):
        super().__init__(name)
        self.kde = None
        self._emp_data = None

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.data.values)
        # This is necessary for conditioning on the data later
        self._emp_data = self.data.copy()
        return()

    def _conditionout(self, keep, remove):
        """Conditions the random variables with name in remove on their domain and marginalizes them out.
        """
        # for name in remove:
        #     # Remove all rows from the data that are outside the domain in name
        #     lower_bound = self.byname(name)['domain'].values()[0]
        #     upper_bound = self.byname(name)['domain'].values()[1]
        #     lower_cond = self.data[name] >= lower_bound
        #     upper_cond = self.data[name] <= upper_bound
        #     self.data = self.data[lower_cond & upper_cond]
        #     # Do the same for the test data
        #     lower_cond = self.test_data[name] >= lower_bound
        #     upper_cond = self.test_data[name] <= upper_bound
        #     self.test_data = self.test_data[lower_cond & upper_cond]
        # self.data = self.data.drop(remove, axis=1)
        # self.test_data = self.test_data.drop(remove, axis=1)

        # collect conditions
        #values = [self.byname(r)['domain'].values() for r in remove]
        #conditions = zip(remove, ['in']*len(values), values)
        # condition
        #self._emp_data = data_op.condition_data(self._emp_data, conditions)


        for name in remove:
            # Remove all rows from the _emp_data that are outside the domain in name
            lower_bound = self.byname(name)['domain'].values()[0]
            upper_bound = self.byname(name)['domain'].values()[1]
            lower_cond = self._emp_data[name] >= lower_bound
            upper_cond = self._emp_data[name] <= upper_bound
            self._emp_data = self._emp_data[lower_cond & upper_cond]

        # transfer condition from _emp_data to data
        self.data = self.data.loc[list(self._emp_data.index.values), :]
        self._marginalizeout(keep, remove)

    def _marginalizeout(self, keep, remove):
        """Marginalizes the dimensions in remove, keeping all those in keep"""
        # Fit the model to the current data. The data dimension to marginalize over
        # should have been removed before
        self._fit()
        return ()

    def _density(self, x):
        """Returns the density at x"""
        x = np.reshape(x, (1, len(x)))
        logdensity = self.kde.score_samples(x)[0]
        return np.exp(logdensity).item()

    def _arithmetic_mean(self):
        """Returns the point of the average density"""
        maximum = data_aggr.aggregate_data(self.data, 'maximum')
        return maximum

    def _sample(self):
        """Returns random point of the distribution"""
        sample = self.kde.sample()
        return sample

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy.kde = copy.deepcopy(self.kde)
        return mycopy


# Create model for testing
if __name__ == "__main__":
    import pandas as pd
    import mb_modelbase as mb

    size = 200
    a = 3*np.random.normal(2, 1, size) + np.random.normal(5, 0.1, size) + np.random.normal(6, 10, size)
    b = np.random.normal(0, 1, size)
    data = pd.DataFrame({'A': a, 'B': b})
    kde_model = mb.KDEModel('kde_model_multimodal_gaussian')
    kde_model.fit(data)
    Model.save(kde_model, '/home/guet_jn/Desktop/mb_data/data_models/kde_model_multimodal_gaussian.mdl')


