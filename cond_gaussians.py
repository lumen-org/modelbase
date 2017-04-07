# Copyright (c) 2017 Philipp Lucas and Frank Nussbaum, FSU Jena

import functools
import logging
import numpy as np
from numpy import nan, pi, exp, dot, abs
from numpy.linalg import inv, det
import xarray as xr

import models as md

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ConditionallyGaussianModel(md.Model):
    """A conditional gaussian model and methods to derive submodels from it
    or query density and other aggregations of it.

    Note that all conditional Gaussians use the same covariance matrix. Their mean, however, is not identical.

    Internal:
        Assume a CG-model on m categorical random variables and n continuous random variables.
        The model is parametrized using the mean parameters as follows:
            _p:
                meaning: probability look-up table for the categorical part of the model.
                data structure used: xarray DataArray. The labels of the dimensions and the coordinates of each
                    dimension are the same as the name of the categorical fields and their domain, respectively.
                    Each entry of the DataArray is a scalar, i.e. the probability of that event.
            _S:
                meaning: covariance matrix of the conditionals. All conditionals share the same covariance!
                data structure used: xarray DataArray with 2 dimensions dim_0 and dim_1. Both dimensions have
                    coordinates that match the names of the continuous random variables.
            _mu:
                meaning: the mean of each conditional gaussian.
                data structure used: xarray DataArray with m+1 dimensions. The first m dimensions represent the
                categorical random variables of the model and are labeled accordingly. The last dimension represents
                the mean vector of the continuous part of the model and is therefore of length n.

            _categoricals:
                list of names of all categorical fields (in same order as they appear in fields)

            _numericals:
                list of names of all continuous fields (in same order as they appear in fields)

            fields:
                list of fields of this model. continuous fields are stored __before__ categorical ones.

        Furthermore there are a number of internally held precomputed values:
            _SInv:
                meaning: the inverse of _S
            _detS
                meaning: the determinant of _S

    Limitations:
        inference queries:
            marginalizing out on any categorical variable leads to an inexact model. Problem is that the best
            CG-approximation does not have a single Sigma but a different one for each remaining x in omega_X.
            See the equation for Sigma_II^"Dach" in the GM-script. We calculate a sum over all categorical
            elements to be removed.
    """

    def __init__(self, name):
        super().__init__(name)

        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        self._categoricals = []
        self._numericals = []
        self._p = xr.DataArray([])
        self._mu = xr.DataArray([])
        self._S = xr.DataArray([])
        self._SInv = nan
        self._detS = nan
        # creates an self contained update function. we use it as a callback function later
        self._unbound_updater = functools.partial(self.__class__._update, self)

    @staticmethod
    def _fitFullLikelihood(data, fields, dc):
        """fit full likelihood for CG model. the data frame data consists of dc many categorical columns and the rest are
        numerical columns. all categorical columns occur before the numerical ones."""
        k = 1  # laplacian smoothing parameter
        n, d = data.shape
        dg = d - dc

        cols = data.columns
        catcols = cols[:dc]
        gausscols = cols[dc:]

        extents = [f['extent'].value() for f in fields[:dc]]  # levels
        sizes = [len(v) for v in extents]
        #        print('extents:', extents)

        z = np.zeros(tuple(sizes))
        pML = xr.DataArray(data=z, coords=extents, dims=catcols)

        # mus
        mus = np.zeros(tuple(sizes + [dg]))
        coords = extents + [[contname for contname in gausscols]]
        dims = list(catcols) + ['mean']
        musML = xr.DataArray(data=mus, coords=coords, dims=dims)

        # calculate p(x)
        for row in data.itertuples():
            cats = row[1:1 + dc]
            gauss = row[1 + dc:]
            pML.loc[cats] += 1
            musML.loc[cats] += gauss

        it = np.nditer(pML, flags=['multi_index'])  # iterator over complete array
        while not it.finished:
            ind = it.multi_index
            musML[ind] /= pML[ind]
            it.iternext()

        # smooth and normalize
        # pML /= 1.0 * n  # without smoothing
        pML = (pML + k) / (pML.sum() + k*pML.size)

        Sigma = np.zeros((dg, dg))
        for row in data.itertuples():
            cats = row[1:1 + dc]
            gauss = row[1 + dc:]
            ymu = gauss - musML.loc[cats]
            Sigma += np.outer(ymu, ymu)

        Sigma /= n
        Sigma = xr.DataArray(Sigma, coords=[gausscols]*2)

        return pML, musML, Sigma

    # base
    def _set_data(self, df, drop_silently):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        """ Internal: estimates the set of mean parameters that fit best to the data given in the
        dataframe df.
        """
        assert(self.mode != 'none')
        df = self.data
        dc = len(self._categoricals)
        self._p, self._mu, self._S = ConditionallyGaussianModel._fitFullLikelihood(df, self.fields, dc)
        return self._unbound_updater,

    def _update(self):
        """Updates dependent parameters / precalculated values of the model after some internal changes."""
        if len(self._numericals) == 0:
            self._detS = nan
            self._SInv = nan
            self._S = xr.DataArray([])
            self._mu = xr.DataArray([])
        else:
            self._detS = abs(det(self._S))
            self._SInv = inv(self._S)

        if len(self._categoricals) == 0:
            self._p = xr.DataArray([])
        return self

    def _conditionout(self, keep, remove):
        remove = set(remove)

        # condition on categorical fields
        cat_remove = [name for name in self._categoricals if name in remove]
        if len(cat_remove) != 0:
            # _S remains unchanged
            pairs = dict(self._condition_values(cat_remove, True))

            # _p changes like in the categoricals.py case
            # trim the probability look-up table to the appropriate subrange and normalize it
            p = self._p.loc[pairs]
            self._p = p / p.sum()

            # _mu is trimmed: keep the slice that we condition on, i.e. reuse the 'pairs' access-structure
            # note: if we condition on all categoricals this also works: it simply remains the single 'selected' mu...
            if len(self._numericals) != 0:
                self._mu = self._mu.loc[pairs]

        # condition on continuous fields
        num_remove = [name for name in self._numericals if name in remove]
        if len(num_remove) == len(self._numericals):
            # all gaussians are removed
            self._S = xr.DataArray([])
            self._mu = xr.DataArray([])
        elif len(num_remove) != 0:
            # collect singular values to condition out
            condvalues = self._condition_values(num_remove)

            # calculate updated mu and sigma for conditional distribution, according to GM script
            j = num_remove  # remove
            i = [name for name in self._numericals if name not in num_remove]  # keep
            S = self._S
            sigma_expr = np.dot(S.loc[i, j], inv(S.loc[j, j]))  # reused below multiple times
            self._S = S.loc[i, i] - dot(sigma_expr, S.loc[j, i])  # upper Schur complement
            cat_keep = self._mu.dims[:-1]
            if len(cat_keep) != 0:
                # iterate over all mu and update them
                # this is a view on mu! it stacks up all categorical dimensions and thus allows us to iterate on them
                stacked = self._mu.stack(pl_stack=cat_keep)
                for coord in stacked.pl_stack:
                    indexer = dict(pl_stack=coord)
                    mu = stacked.loc[indexer]
                    # extent indexer to subselect only the part of mu that is updated. the rest is removed later.
                    # problem is: we cannot assign a shorter vector to stacked.loc[indexer]
                    indexer['mean'] = i
                    stacked.loc[indexer] = mu.loc[i] + dot(sigma_expr, condvalues - mu.loc[j])
                # above we partially updated only the relevant part of mu. the remaining part is now removed:
                self._mu = self._mu.loc[dict(mean=i)]
            else:
                # special case: no categorical fields left. hence we cannot stack over then, it is only a single mu left
                # and we only need to update that
                self._mu = self._mu.loc[i] + dot(sigma_expr, condvalues - self._mu.loc[j])

        # remove fields as needed
        self._categoricals = [name for name in self._categoricals if name not in remove]
        self._numericals = [name for name in self._numericals if name not in remove]
        return self._unbound_updater,

    def _marginalizeout(self, keep, remove):
        # use weak marginals to get the best approximation of the marginal distribution that is still a cg-distribution
        keep = set(keep)
        num_keep = [name for name in self._numericals if name in keep]  # note: this is guaranteed to be sorted
        cat_remove = [name for name in self._categoricals if name not in keep]

        if len(self._categoricals) != 0:  # only enter if there is work to do
            # clone old p for later reuse
            p = self._p.copy()
            # marginalized p: just like in the categorical case (categoricals.py), i.e. sum up over removed dimensions
            self._p = self._p.sum(cat_remove)

        # marginalized mu (taken from the script)
        # slice out the gaussian part to keep; sum over the categorical part to remove
        if len(num_keep) != 0:
            mu = self._mu.loc[dict(mean=num_keep)]
            if len(self._categoricals) == 0:
                self._mu = mu
            else:
                self._mu = (p * mu).sum(cat_remove) / self._p
            # marginalized sigma
            self._S = self._S.loc[num_keep, num_keep]

        # update fields and dependent variables
        self._categoricals = [name for name in self._categoricals if name in keep]
        self._numericals = num_keep
        return self._unbound_updater,

    def _density(self, x):
        cat_len = len(self._categoricals)
        num_len = len(self._numericals)
        cat = tuple(x[:cat_len])  # need it as a tuple for indexing below
        num = np.array(x[cat_len:])  # need as np array for dot product

        p = self._p.loc[cat].data

        if num_len == 0:
            return p

        # works because gaussian variables are - by design of this class - after categoricals.
        # Therefore the only not specified dimension is the last one, i.e. the one that holds the mean!
        mu = self._mu.loc[cat].data

        xmu = num - mu
        gauss = (2 * pi) ** (-num_len / 2) * (self._detS ** -.5) * exp(-.5 * np.dot(xmu, np.dot(self._SInv, xmu)))

        if cat_len == 0:
            return gauss
        else:
            return p * gauss

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        cat_len = len(self._categoricals)
        num_len = len(self._numericals)

        if cat_len == 0:
            # then there is only a single gaussian left and the maximum is its mean value, i.e. the value of _mu
            return list(self._mu.data)

        # categorical part
        # I think its just scanning over all x of omega_x, since there is only one sigma
        # i.e. this is like in the categorical case
        p = self._p
        pmax = p.where(p == p.max(), drop=True)  # get view on maximum (coordinates remain)
        cat_argmax = [idx[0] for idx in pmax.indexes.values()]  # extract coordinates from indexes

        if num_len == 0:
            return cat_argmax

        # gaussian part
        # the mu is the mean and the maximum of the conditional gaussian
        num_argmax = self._mu.loc[tuple(cat_argmax)]
        return cat_argmax + list(num_argmax.data)

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._mu = self._mu.copy()
        mycopy._S = self._S.copy()
        mycopy._p = self._p.copy()
        mycopy._categoricals = self._categoricals.copy()
        mycopy._numericals = self._numericals.copy()
        mycopy._update()
        return mycopy

if __name__ == '__main__':
    pass
