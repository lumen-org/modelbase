# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)

import logging

from mb_modelbase.core.fixed_mixture_model import FixedMixtureModel
from mb_modelbase.core.cond_gaussian_wm import CgWmModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MixtureOfCgWmModel(FixedMixtureModel):
    """A mixture of Conditional Gaussians (CG) with Weak Marginals (WM).

    This mixture of CG with WM has a fixed number of mixture components, which is set at construction time by the k parameter.

    The point here is the following: CGs are not closed under marginalization of discrete random variables. As you can easily verify the true marginals are mixtures of CG. However,these mixtures have a 'dynamic number of mixture components': the more discrete variables you marginalize, the more mixture components you get.

    This model type in turn has a fixed number of components, and each component is handled independently just the way a single CG with WM is handled. The number of components therefore does not increase when marginalizing over discrete random variables.
    """

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

    _maximum = FixedMixtureModel._maximum_naive_heuristic

def MoCGModelWithK(name, k):
    """Returns an empty Mixture of Cond Gaussians model."""
    model = MixtureOfCgWmModel(name)
    model.set_k(k)
    return model


if __name__ == "__main__":
    import data.iris.iris as iris
    data = iris.mixed('data/iris/iris.csv')
    model = MixtureOfCgWmModel("my mixture model")
    model.set_k(2)
    model.fit(data)
    model.mode = "model"
    print(str(model.density(values=[1, 2, 3, 4])))
