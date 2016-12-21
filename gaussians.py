# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
import copy as cp
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

    @staticmethod
    def _get_header(df):
        """ Returns suitable fields for a model from a given pandas dataframe.
        """
        fields = []
        for column in df:
            field = md.Field(column, dm.NumericDomain(), dm.NumericDomain(df[column].min(), df[column].max()), 'numerical')
            fields.append(field)
        return fields

    def fit(self, df):
        """Fits the model to passed DataFrame

        Returns:
            The modified model.
        """
        self.data = df
        self.fields = MultiVariateGaussianModel._get_header(self.data)

        # fit using scikit learn mixtures
        model = mixture.GMM(n_components=1, covariance_type='full')
        model.fit(self.data)
        self._mu = matrix(model.means_).T
        self._S = matrix(model.covars_)
        return self.update()

    def __str__(self):
        return ("Multivariate Gaussian Model '" + self.name + "':\n" +
                "dimension: " + str(self._n) + "\n" +
                "names: " + str([self.names]) + "\n" +
                "fields: " + str([str(field) for field in self.fields]))

    def update(self):
        """updates dependent parameters / precalculated values of the model"""
        self._update()
        if self._n == 0:
            self._detS = nan
            self._SInv = nan
        else:
            self._detS = np.abs(np.linalg.det(self._S))
            self._SInv = self._S.I

        assert (self._mu.shape == (self._n, 1) and
                self._S.shape == (self._n, self._n))

        return self

    def _conditionout(self, remove):
        """Conditions the random variables with name in remove on their available, //not unbounded// domain and marginalizes
                them out.

                Note that we don't know yet how to condition on a non-singular domain (i.e. condition on interval or sets).
                As a work around we therefore:
                  * for continuous domains: condition on (high-low)/2
                  * for discrete domains: condition on the first element in the domain
         """

        if len(remove) == 0 or self._isempty():
            return self
        if len(remove) == self._n:
            return self._setempty()

        # collect singular values to condition out on
        j = sorted(self.asindex(remove))
        condvalues = []
        for idx in j:
            field = self.fields[idx]
            domain = field['domain']
            dvalue = domain.value()
            assert (domain.isbounded())
            if field['dtype'] == 'numerical':
                condvalues.append(dvalue if domain.issingular() else (dvalue[1] - dvalue[0]) / 2)
                # TODO: we don't know yet how to condition on a not singular, but not unrestricted domain.
            else:
                raise ValueError('invalid dtype of field: ' + str(field['dtype']))
        condvalues = matrix(condvalues).T

        # calculate updated mu and sigma for conditional distribution, according to GM script
        i = utils.invert_indexes(j, self._n)
        S = self._S
        mu = self._mu
        self._S = MultiVariateGaussianModel._schurcompl_upper(S, i)
        self._mu = mu[i] + S[ix_(i, j)] * S[ix_(j, j)].I * (condvalues - mu[j])

        self.fields = [self.fields[idx] for idx in i]
        return self.update()

    def _marginalizeout(self, keep):
        if len(keep) == self._n or self._isempty():
            return self
        if len(keep) == 0:
            return self._setempty()

        # i.e.: just select the part of mu and sigma that remains
        keepidx = sorted(self.asindex(keep))
        self._mu = self._mu[keepidx]
        self._S = self._S[np.ix_(keepidx, keepidx)]
        self.fields = [self.fields[idx] for idx in keepidx]
        return self.update()

    def _density(self, x):
        """Returns the density of the model at point x.

        Args:
            OLD: x: a Scalar or a _column_ vector as a numpy matrix.
            x: a list of values as input for the density.
        """
        x = matrix(x).T  # turn into column vector of type numpy matrix
        xmu = x - self._mu
        return ((2 * pi) ** (-self._n / 2) * (self._detS ** -.5) * exp(-.5 * xmu.T * self._SInv * xmu)).item()

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        # _mu is a np matrix, but we want to return a list
        return self._mu.T.tolist()[0]

    def _sample(self):
        # TODO: let it return a dataframe?
        return self._S * np.matrix(np.random.randn(self._n)).T + self._mu

    def copy(self, name=None):
        # TODO: this should be as lightweight as possible!
        name = self.name if name is None else name
        mycopy = MultiVariateGaussianModel(name)
        mycopy.data = self.data
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy._mu = self._mu.copy()
        mycopy._S = self._S.copy()
        mycopy.update()
        return mycopy

    @staticmethod
    def custom_mvg(sigma, mu, name):
        """Returns a MultiVariateGaussian model that uses the provided sigma, mu and name.

        Note: The domain of each field is set to (-10,10).

        Args:
            sigma: a suitable numpy matrix
            mu: a suitable numpy row vector
        """

        if not isinstance(mu, matrix) or not isinstance(sigma, matrix) or mu.shape[1] != 1:
            raise ValueError("invalid arguments")
        model = MultiVariateGaussianModel(name)
        model._S = sigma
        model._mu = mu
        model.fields = [md.Field(name="dim" + str(idx),
                                 domain=dm.NumericDomain(),
                                 extent=dm.NumericDomain(mu[idx].item() - 2, mu[idx].item() + 2))
                        for idx in range(sigma.shape[0])]
        model.update()
        return model

    @staticmethod
    def normal_mvg(dim, name):
        sigma = matrix(np.eye(dim))
        mu = matrix(np.zeros(dim)).T
        return MultiVariateGaussianModel.custom_mvg(sigma, mu, name)

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


if __name__ == '__main__':
    import pdb

    # foo = MultiVariateGaussianModel.normalMVG(5,"foo")
    sigma = matrix([
        [1.0, 0.6, 0.0, 2.0],
        [0.6, 1.0, 0.4, 0.0],
        [0.0, 0.4, 1.0, 0.0],
        [2.0, 0.0, 0.0, 1.]])
    mu = np.matrix([1.0, 2.0, 0.0, 0.5]).T
    foo = MultiVariateGaussianModel.custom_mvg(sigma, mu, "foo")
    foocp = foo.copy("foocp")
    print("\n\nmodel 1\n" + str(foocp))
    foocp2 = foocp.model(['dim1', 'dim0'], as_="foocp2")
    print("\n\nmodel 2\n" + str(foocp2))

    res = foo.predict(predict=['dim0'], splitby=[SplitTuple('dim0', 'equiDist', [5])])
    print("\n\npredict 1\n" + str(res))

    res = foo.predict(predict=[AggregationTuple(['dim1'], 'maximum', 'dim1', []), 'dim0'],
                      splitby=[SplitTuple('dim0', 'equiDist', [10])])
    print("\n\npredict 2\n" + str(res))

    res = foo.predict(predict=[AggregationTuple(['dim0'], 'maximum', 'dim0', []), 'dim0'],
                      where=[ConditionTuple('dim0', 'equals', 1)], splitby=[SplitTuple('dim0', 'equiDist', [10])])
    print("\n\npredict 3\n" + str(res))

    res = foo.predict(predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
                      splitby=[SplitTuple('dim0', 'equiDist', [10])])
    print("\n\npredict 4\n" + str(res))

    res = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        where=[ConditionTuple('dim0', 'greater', -1)])
    print("\n\npredict 5\n" + str(res))

    res = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        where=[ConditionTuple('dim0', 'less', -1)])
    print("\n\npredict 6\n" + str(res))

    res = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        where=[ConditionTuple('dim0', 'less', 0), ConditionTuple('dim2', 'equals', -5.0)])
    print("\n\npredict 7\n" + str(res))

    res, base = foo.predict(
        predict=[AggregationTuple(['dim0'], 'density', 'dim0', []), 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10]), SplitTuple('dim1', 'equiDist', [7])],
        where=[ConditionTuple('dim0', 'less', -1), ConditionTuple('dim2', 'equals', -5.0)],
        returnbasemodel=True)
    print("\n\npredict 8\n" + str(res))

    res, base = foo.predict(
        predict=[AggregationTuple(['dim0'], 'average', 'dim0', []), AggregationTuple(['dim0'], 'density', 'dim0', []),
                 'dim0'],
        splitby=[SplitTuple('dim0', 'equiDist', [10])],
        # where=[ConditionTuple('dim0', 'less', -1), ConditionTuple('dim2', 'equals', -5.0)],
        returnbasemodel=True)
    print("\n\npredict 9\n" + str(res))

    res, base = foo.predict(
        predict=['dim0', 'dim1', AggregationTuple(['dim0', 'dim1'], 'average', 'dim0', []),
                 AggregationTuple(['dim0', 'dim1'], 'average', 'dim1', [])],
        splitby=[SplitTuple('dim0', 'identity', []), SplitTuple('dim1', 'equiDist', [4])],
        where=[ConditionTuple('dim0', '<', 2), ConditionTuple('dim0', '>', 1)],
        returnbasemodel=True)
    print("\n\npredict 10\n" + str(res))

    # print("\n\n" + str(base) + "\n")