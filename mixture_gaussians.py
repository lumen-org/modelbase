# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)

import logging

from fixed_mixture_model import FixedMixtureModel
from gaussians import MultiVariateGaussianModel

from sklearn import mixture
from numpy import matrix

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MixtureOfGaussiansModel(FixedMixtureModel):
    """A mixture of Gaussians Model."""

    def __init__(self, name):
        super().__init__(name)
        self._k = None
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }

    def set_k(self, k):
        self._k = k
        self._set_models([(MultiVariateGaussianModel, k)])

    def _set_data_4mixture(self, df, drop_silently):
        self._set_data_continuous(df, drop_silently)
        return ()

    def _fit(self):
        # learn model using sklearn
        sklgmm = mixture.GMM(n_components=self._k, covariance_type='full')
        sklgmm.fit(self.data)
        mus = sklgmm.means_
        Sigmas = sklgmm.covars_
        weights = sklgmm.weights_

        # set mean and covar of each component
        for idx, (weight, model) in enumerate(zip(self.weights, self)):
            model._mu = matrix(mus[idx]).T
            model._S = matrix(Sigmas[idx])
            self.weights[idx] = weights[idx]
            model._update()  # stupid me!!
        return self._unbound_component_updater,

    #_maximum = FixedMixtureModel._maximum_naive_heuristic
    _maximum = FixedMixtureModel._maximum_better_heuristic




def MoGModelWithK(name, k):
    """Returns an empty Mixture of k Gaussians model."""
    model = MixtureOfGaussiansModel(name)
    model.set_k(k)
    return model


if __name__ == "__main__":
#    import data.iris.iris as iris
#    iris = iris.mixed('data/iris/iris.csv')
#    model = MixtureOfGaussiansModel("my mixture model")
#    model.set_k(3)
#    model.fit(iris)
#    model.mode = "model"
#    print(str(model.density([1, 2, 3, 4])))
    pass