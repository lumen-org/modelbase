# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
from sklearn.neighbors.kde import KernelDensity

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

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.data)
        return()

    def _conditionout(self, keep, remove):
        """Conditions the random variables with name in remove on their domain and marginalizes them out.
        """

    def _marginalizeout(self, keep, remove):
        """Marginalizes the dimensions in remove, keeping all those in keep"""

        return ()