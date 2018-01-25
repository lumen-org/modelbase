# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import functools
import logging
import numpy as np
from numpy import pi, exp, matrix, ix_, nan
from sklearn import mixture

from mb_modelbase.utils import utils
from mb_modelbase.models_core import models as md
from mb_modelbase.models_core import domains as dm

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class MultiVariateGaussianModel(md.Model):
    """A multivariate gaussian model and methods to derive submodels from it
    or query density and other aggregations of it
    """

    def __init__(self, name):
        super().__init__(name)
        self._mu = nan
        self._S = nan
        self._detS = nan
        self._SInv = nan
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        # creates an self contained update function. we use it as a callback function later
        self._unbound_updater = functools.partial(self.__class__._update, self)

    def _set_data(self, df, drop_silently):
        self._set_data_continuous(df, drop_silently)
        return ()

    def _fit(self):
        """Fits the model to data of the model

        Returns:
            The modified model.
        """

        # fit using scikit learn mixtures
        model = mixture.GMM(n_components=1, covariance_type='full')
        model.fit(self.data)
        self._mu = matrix(model.means_).T
        self._S = matrix(model.covars_)
        return self._unbound_updater,

    def __str__(self):
        return ("Multivariate Gaussian Model '" + self.name + "':\n" +
                "dimension: " + str(self.dim) + "\n" +
                "names: " + str([self.names]) + "\n" +
                "fields: " + str([str(field) for field in self.fields]))

    def _update(self):
        """updates dependent parameters / precalculated values of the model"""
        if self.dim == 0:
            self._detS = nan
            self._SInv = nan
        else:
            self._detS = np.abs(np.linalg.det(self._S))
            self._SInv = self._S.I

        assert (self._mu.shape == (self.dim, 1) and
                self._S.shape == (self.dim, self.dim))

        return self

    def _conditionout(self, keep, remove):
        """Conditions the random variables with name in remove on their available, //not unbounded// domain and marginalizes
                them out.

                Note that we don't know yet how to condition on a non-singular domain (i.e. condition on interval or sets).
                As a work around we therefore:
                  * for continuous domains: condition on (high-low)/2
                  * for discrete domains: condition on the first element in the domain
         """
        # collect singular values to condition out on
        condvalues = self._condition_values(remove)
        condvalues = matrix(condvalues).T

        # calculate updated mu and sigma for conditional distribution, according to GM script
        j = self.asindex(remove)
        i = utils.invert_indexes(j, self.dim)
        S = self._SInv
        mu = self._mu
        self._S = utils.schur_complement(S, i)
        self._mu = mu[i] + S[ix_(i, j)] * S[ix_(j, j)].I * (condvalues - mu[j])

        return self._unbound_updater,

    def _marginalizeout(self, keep, remove):
        # i.e.: just select the part of mu and sigma that remains
        keepidx = self.asindex(keep)
        self._mu = self._mu[keepidx]
        self._S = self._S[np.ix_(keepidx, keepidx)]
        return self._unbound_updater,

    def _density(self, x):
        x = matrix(x).T  # turn into column vector of type numpy matrix
        xmu = x - self._mu
        return ((2 * pi) ** (-self.dim / 2) * (self._detS ** -.5) * exp(-.5 * xmu.T * self._SInv * xmu)).item()

    def _gradient(self, x):
        x = matrix(x).T 
        xmu = x - self._mu
        result = ((2 * pi) ** (-self.dim / 2) * (self._detS ** -.5) * exp(-.5 * xmu.T * self._SInv * xmu)).item() * (-.5 * self._SInv * xmu)
        return np.array(result).T[0]

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        # _mu is a np matrix, but we want to return a list
        return self._mu.T.tolist()[0]

    def _sample(self):
        sample = self._S * np.matrix(np.random.randn(self.dim)).T + self._mu
        return sample.T.tolist()[0]   # we want it as a 'normal' list

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._mu = self._mu.copy()
        mycopy._S = self._S.copy()
        mycopy._update()
        return mycopy

    def _generate_model(self, opts):
        """Generates a gaussian model according to options. This does not assign any data to the model.
        It also sets the 'mode' of this model to 'model'.

        Note that this method is solely called by Model.generate_model.

        Options must have a key 'mode' of value 'custom' or 'normal':

        If mode is 'custom':
            option must have keys 'sigma' and 'mu'. which are a suitable numpy matrix and row vector, resp.
            The domain of each field is set to (-10,10).

        if mode is 'normal'
            option must have a key 'dim'. its value is the dimension of the model. The model will have the
            identity matrix as its sigma and the zero vector as its mean.

        """

        mode = opts['mode']
        if mode == 'custom':
            mu = opts['mu']
            sigma = opts['sigma']
            if not isinstance(mu, matrix) or not isinstance(sigma, matrix) or mu.shape[1] != 1:
                raise ValueError("invalid arguments")
            self._S = sigma
            self._mu = mu
            self.fields = [md.Field(name="dim" + str(idx),
                                     domain=dm.NumericDomain(),
                                     extent=dm.NumericDomain(mu[idx].item() - 2, mu[idx].item() + 2))
                           for idx in range(sigma.shape[0])]
            self.mode = 'model'
            return self._unbound_updater,

        if mode == 'normal':
            dim = opts['dim']
            sigma = matrix(np.eye(dim))
            mu = matrix(np.zeros(dim)).T
            opts = {'mode': 'custom', 'sigma': sigma, 'mu': mu}
            return self._generate_model(opts=opts)
        else:
            raise ValueError('invalid mode: ' + str(mode))


    @staticmethod
    def dummy2d_model(name='test'):
        m = MultiVariateGaussianModel(name)
        mu = np.matrix(np.zeros(2)).T
        sigma = np.matrix([[1, 0.5], [0.5, 1]])
        m.generate_model({'mode': 'custom', 'mu': mu, 'sigma': sigma})
        m._generate_data({'n': 200})
        return m

if __name__ == '__main__':
    import pdb

