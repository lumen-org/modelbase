# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import functools
import logging
import math

from fixed_mixture_model import FixedMixtureModel
from gaussians import MultiVariateGaussianModel
import seaborn.apionly as sns

from sklearn import mixture
from numpy import pi, exp, matrix, ix_, nan

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
        return self._unbound_component_updater,

    def _fit(self):
        # learn model using sklearn
        sklgmm = mixture.GMM(n_components=self._k, covariance_type='full')
        sklgmm.fit(self.data)
        mus = sklgmm.means_
        Sigmas = sklgmm.covars_

        # set mean and covar of each component
        for idx, model in enumerate(self):
            model._mu = matrix(mus[idx]).T
            model._S = matrix(Sigmas[idx])
            model._update()
        return self._unbound_component_updater,

    def _maximum(self):
        # this is an pretty stupid heuristic :-)
        maximum = None
        maximum_density = -math.inf

        for model in self:
            cur_maximum = model._maximum()
            cur_density = model._density(cur_maximum)
            if cur_density > maximum_density:
                maximum = cur_maximum
                maximum_density = cur_density

        return maximum

    def _generate_model(self, opts={}):
        # TODO
        """opts has a optional key 'dim' that defaults to 6 and specifies the dimension of the model that you want to
        have."""
        # if 'dim' not in opts:
        #     dim = 6
        # else:
        #     dim = opts['dim']
        #
        # ncat = math.floor(dim/2)
        # nnum = dim - ncat
        # self.fields = []
        #
        # # numeric
        # for idx in range(ncat):
        #     field = md.Field(name="dim" + str(idx),
        #                      domain=dm.NumericDomain(),
        #                      extent=dm.NumericDomain(-5, 5))
        #     self.fields.append(field)
        #
        # # categorical
        # for idx in range(ncat):
        #     field = md.Field(name="dim" + str(idx+nnum),
        #                      domain=dm.DiscreteDomain(),
        #                      extent=dm.DiscreteDomain(list("ABCDEFG")))
        #     self.fields.append(field)
        #
        # self.mode = 'model'
        # return MixtureOfGaussiansModel.update,
        raise NotImplementedError("Implement this method in your subclass")


def MoGModelWithK(name, k):
    """Returns an empty Mixture of k Gaussians model"""
    model = MixtureOfGaussiansModel(name)
    model.set_k(k)
    return model


if __name__ == "__main__":
    iris = sns.load_dataset('iris').iloc[:, 0:-1]
    model = MixtureOfGaussiansModel("my mixture model")
    model.set_k(3)
    model.fit(iris)
    model.mode = "model"
    print(str(model.density([1, 2, 3, 4])))
    pass