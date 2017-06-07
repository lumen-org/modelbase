# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)

import logging

from fixed_mixture_model import FixedMixtureModel
from cond_gaussian_wm import CgWmModel
import seaborn.apionly as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MixtureOfCgWmModel(FixedMixtureModel):
    """A mixture of Conditional Gaussians with Weak Marginals."""

    def __init__(self, name):
        super().__init__(name)
        self._k = None
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }

    def set_k(self, k):
        self._k = k
        self._set_models([(CgWmModel, k)])

    def _set_data_4mixture(self, df, drop_silently):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        # TODO!
        raise NotImplementedError

        # set mean and covar of each component
        # for idx, (weight, model) in enumerate(zip(self.weights, self)):
        #     model._mu = matrix(mus[idx]).T
        #     model._S = matrix(Sigmas[idx])
        #     self.weights[idx] = weights[idx]
        #     model._update()  # stupid me!!
        # return self._unbound_component_updater,

    _maximum = FixedMixtureModel._maximum_naiv_heuristic

def MoCGModelWithK(name, k):
    """Returns an empty Mixture of Cond Gaussians model."""
    model = MixtureOfCgWmModel(name)
    model.set_k(k)
    return model


if __name__ == "__main__":
    iris = sns.load_dataset('iris')
    model = MixtureOfCgWmModel("my mixture model")
    model.set_k(2)
    model.fit(iris)
    model.mode = "model"
    print(str(model.density([1, 2, 3, 4])))
    pass