# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import logging
import numpy as np
from numpy import pi, exp, matrix, ix_, nan
from sklearn import mixture

import utils
import models as md
from models import AggregationTuple, SplitTuple, ConditionTuple
import domains as dm

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

    def _set_data(self, df, drop_silently):
        return self._set_data_continuous(df, drop_silently)

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
        return self.update()

    def __str__(self):
        return ("Multivariate Gaussian Model '" + self.name + "':\n" +
                "dimension: " + str(self.dim) + "\n" +
                "names: " + str([self.names]) + "\n" +
                "fields: " + str([str(field) for field in self.fields]))

    def update(self):
        """updates dependent parameters / precalculated values of the model"""
        self._update()
        if self.dim == 0:
            self._detS = nan
            self._SInv = nan
        else:
            self._detS = np.abs(np.linalg.det(self._S))
            self._SInv = self._S.I

        assert (self._mu.shape == (self.dim, 1) and
                self._S.shape == (self.dim, self.dim))

        return self

    def _conditionout(self, remove):
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
        S = self._S
        mu = self._mu
        self._S = MultiVariateGaussianModel._schurcompl_upper(S, i)
        self._mu = mu[i] + S[ix_(i, j)] * S[ix_(j, j)].I * (condvalues - mu[j])

        self.fields = [self.fields[idx] for idx in i]
        return self.update()

    def _marginalizeout(self, keep):
        # i.e.: just select the part of mu and sigma that remains
        keepidx = self.asindex(keep)
        self._mu = self._mu[keepidx]
        self._S = self._S[np.ix_(keepidx, keepidx)]
        self.fields = [self.fields[idx] for idx in keepidx]
        return self.update()

    def _density(self, x):
        x = matrix(x).T  # turn into column vector of type numpy matrix
        xmu = x - self._mu
        return ((2 * pi) ** (-self.dim / 2) * (self._detS ** -.5) * exp(-.5 * xmu.T * self._SInv * xmu)).item()

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
        mycopy.update()
        return mycopy

    def _generate_model(self, opts):
        """Generates a gaussian model according to options. This does not assign any data to the model.
        It also sets the 'mode' of this model to 'model'.

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
            self.update()
            return self

        if mode == 'normal':
            dim = opts['dim']
            sigma = matrix(np.eye(dim))
            mu = matrix(np.zeros(dim)).T
            opts = {'mode': 'custom', 'sigma': sigma, 'mu': mu}
            return self._generate_model(opts=opts)
        else:
            raise ValueError('invalid mode: ' + str(mode))


    @staticmethod
    def _schurcompl_upper(M, idx):
        """Returns the upper Schur complement of array_like M with the 'upper block'
        indexed by idx.
        """
        M = matrix(M, copy=False)  # matrix view on M
        # derive index lists
        i = idx
        j = utils.invert_indexes(i, M.shape[0])
        # that's the definition of the upper Schur complement
        return M[ix_(i, i)] - M[ix_(i, j)] * M[ix_(j, j)].I * M[ix_(j, i)]

    @staticmethod
    def dummy2d_model(name='test'):
        m = MultiVariateGaussianModel(name)
        mu = np.matrix(np.zeros(2)).T
        sigma = np.matrix([[1, 0.5], [0.5, 1]])
        m._generate_model({'mode': 'custom', 'mu': mu, 'sigma': sigma})
        m._generate_data({'n': 200})
        return m

if __name__ == '__main__':
    import pdb

