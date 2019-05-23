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

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.data.values)
        return()

    def _conditionout(self, keep, remove):
        """Conditions the random variables with name in remove on their domain and marginalizes them out.
        """
        for name in remove:
            # Remove all rows from the data that are outside the domain in name
            lower_bound = self.byname(name)['domain'].values()[0]
            upper_bound = self.byname(name)['domain'].values()[1]
            lower_cond = self.data[name] >= lower_bound
            upper_cond = self.data[name] <= upper_bound
            self.data = self.data[lower_cond & upper_cond]
            # Do the same for the test data
            lower_cond = self.test_data[name] >= lower_bound
            upper_cond = self.test_data[name] <= upper_bound
            self.test_data = self.test_data[lower_cond & upper_cond]
        self.data = self.data.drop(remove, axis=1)
        self.test_data = self.test_data.drop(remove, axis=1)
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

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy.kde = copy.deepcopy(self.kde)
        return mycopy