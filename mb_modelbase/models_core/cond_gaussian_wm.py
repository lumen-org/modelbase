# Copyright (c) 2017 Philipp Lucas and Frank Nussbaum, FSU Jena
import functools
import logging
import numpy as np
from numpy import nan, pi, exp, dot, abs, ix_
from numpy.linalg import inv, det
import xarray as xr

import mb_modelbase.utils as utils
from mb_modelbase.utils import no_nan
from mb_modelbase.models_core import models as md
from mb_modelbase.models_core import cond_gaussian_fitting as cgf
from mb_modelbase.models_core import cond_gaussians as cg

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def numeric_extents_from_params(S, mu, nums, factor=3):
    """Returns a factor * stddev extent of the conditional gaussian given bay xarray covariance matrix and xarray
    mean.
    """
    mus = mu.values.reshape(-1, len(nums))
    Ss = S.values.reshape(-1, len(nums), len(nums))
    diags = np.sqrt(np.array([np.diag(s) for s in Ss])) * factor
    mins = (mus - diags).min(axis=0)
    maxs = (mus + diags).max(axis=0)
    return list(zip(mins, maxs))


def common_sigma_to_full_sigma(p, mu, Sigma, numericals):
    """Transforms a set of mean parameters with common sigma (as generated for class cond_gaussian) to the equivalent
     general mean parameters in xarray representation.

     p, mu and Sigma are xarray DataArrays with correct dimensions and coordinates.
     """

    # replicate S_single to a individual S for each cg
    # setup coords as dict of dim names to extents
    dims = p.dims
    coords = dict(p.coords)
    coords['S1'] = numericals  # extent is the list of numerical variables
    coords['S2'] = numericals
    sizes = [len(coords[dim]) for dim in dims]  # generate blow-up sizes

    S = np.outer(np.ones(tuple(sizes)), Sigma.values)  # replicate S as many times as needed
    dims += ('S1', 'S2')  # add missing dimensions for Sigma
    shape = tuple(sizes + [len(numericals)] * 2)  # update shape
    S = S.reshape(shape)  # reshape to match dimension requirements

    return p, mu, xr.DataArray(data=S, coords=coords, dims=dims)


def fit_full(data, fields, categoricals, numericals):
    """Fits a conditional gaussian model to given data and returns the fitting parameters as a 3-tuple(p, mu, S, mu):

        p, mu and S are just like the CgWmModel expects them (see class documentation there)

    Note, however that this returns the same covariance for each conditional gaussian.
    This will be changed in the future!
    """
    dc = len(categoricals)
    p, mu, Sigma = cg.ConditionallyGaussianModel._fitFullLikelihood(data, fields, dc)
    return common_sigma_to_full_sigma(p, mu, Sigma, numericals)


def fit_CLZ(df, categoricals, numericals):
    """Fits a conditional gaussian model to given data and returns the fitting parameters as a 3-tuple(p, mu, S, mu):
        p, mu and S are just like the CgWmModel expects them (see class documentation there)

    Note that this actually is a CLZ triple interaction model.
    """
    (p, mu, Sigma, meta) = cgf.fit_clz_mean(df)
    return utils.numpy_to_xarray_params(p, mu, Sigma, [meta['catuniques'][cat] for cat in categoricals], categoricals, numericals)


def fit_MAP(df, categoricals, numericals):
    """Fits a conditional gaussian model to given data and returns the fitting parameters as a 3-tuple(p, mu, S, mu):
        p, mu and S are just like the CgWmModel expects them (see class documentation there)

    Note that this actually is a CLZ triple interaction model.
    """
    (p, mu, Sigma, meta) = cgf.fit_map_mean(df)
    return utils.numpy_to_xarray_params(p, mu, Sigma, [meta['catuniques'][cat] for cat in categoricals], categoricals, numericals)

def _maximum_cgwm_heuristic1(cat_len, num_len, mu, p, detS):
    """Returns an approximation to point of the maximum density of a given cg distribution.
    Essentially its the coordinates of the most likeliest mean of all gaussians in the cg distribution.

     observation 1: for a given x in Omega_X the maximum is taken at the corresponding cg's mu, lets call it
     argmu(x). Hence, in order to determine the maximum, we scan over all x of Omega_x and calculate the density
    over p(x,argmu(x))

     observation 2: the density of a gaussian at its mean is quite simple since the (x-mu) terms evaluate to 0.

     observation 3: we are only interested in where the maximum is taken, not its actual value. Hence we can remove
     any values that are equal for all, i.e. normalization terms.

     Using observation 2 and 3, the following simplifies:

         (2*pi)^(-n/2) * det(Sigma)^-0.5 * exp( -0.5 * (x-mu)^T * Sigma^-1 * (x-mu) )

     to:

         det(Sigma)^-0.5

     observation 4: luckily, we already have that precalculated!
     """
    if cat_len == 0:
        # then there is only a single gaussian left and the maximum is its mean value, i.e. the value of _mu
        return list(mu.values)

    if num_len == 0:
        # find maximum in p and return its coordinates
        pmax = p.where(p == p.max(), drop=True)  # get view on maximum (coordinates remain)
        return [idx[0] for idx in pmax.indexes.values()]  # extract coordinates from indexes

    else:
        # find compound maximum
        # compute pseudo-density at all gaussian means to find maximum
        pd = p * detS
        pdmax = pd.where(pd == pd.max(), drop=True)  # get view on maximum (coordinates remain)

        # now figure out the coordinates
        cat_argmax = [idx[0] for idx in pdmax.indexes.values()]  # extract categorical coordinates from indexes
        num_argmax = mu.loc[tuple(cat_argmax)]  # extract numerical coordinates as mean

        # return compound coordinates
        return cat_argmax + list(num_argmax.values)


class CgWmModel(md.Model):
    """A conditional gaussian model and methods to derive submodels from it or query density and other aggregations of
     it.

    In this model each conditional Gaussian uses its own covariance matrix and mean vector. This is
    the most important difference to the ConditionallyGaussianModel class.

    Internal:
        Assume a CG-model on m categorical random variables and n continuous random variables.
        The model is parametrized using mean parameters as follows:
            _p:
                meaning: probability look-up table for the categorical part of the model.
                data structure used: xarray DataArray. The labels of the dimensions and the coordinates of each
                    dimension are the same as the name of the categorical fields and their domain, respectively.
                    Each entry of the DataArray is a scalar, i.e. the probability of that event.

            _S:
                meaning: covariance matrix of each of the conditionals.
                data structure used: xarray DataArray with m+2 dimensions. The first m dimensions represent the
                    categorical random variables of the model and are labeled accordingly. The last two dimensions
                    represents the mean vector of the continuous part of the model and are therefore of length n.
                    The labels of the last two dimensions are 'S1' and 'S2'.

            _mu:
                meaning: the mean of each conditional gaussian.
                data structure used: xarray DataArray with m+1 dimensions. The first m dimensions represent the
                    categorical random variables of the model and are labeled accordingly. The last dimension represents
                    the mean vector of the continuous part of the model and is therefore of length n. The label of the
                    last dimension is 'mean'

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
                meaning: abs(det(S))**-0.5

    Limitations:
        inference queries:
            Marginalizing out on any categorical variable leads to an inexact model. Conditional Gaussian models are
             not closed under marginalization, but lead to a mixture of conditional gaussians. In this model we stay
             inside the class of CG Models by using the best CG Model (in terms of the Kulbeck-Leibler divergence)
             as an approximation to the true marginal model. This is called weak marginals (WM) and gave the name
             to this class.
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
        self._SInv = xr.DataArray([])
        self._detS = xr.DataArray([])
        # creates an self contained update function. we use it as a callback function later
        self._unbound_updater = functools.partial(self.__class__._update, self)

    # base
    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        assert (self.mode != 'none')
        self._p, self._mu, self._S = fit_full(self.data, self.fields, self._categoricals, self._numericals)
        for o in [self._p, self._mu, self._S]:
            assert(no_nan(o))
        return self._unbound_updater,

    def _assert_no_nans(self):
        for o in [self._detS, self._SInv, self._S, self._mu, self._p]:
            assert (not np.isnan(o).any())

    def _update(self):
        """Updates dependent parameters / precalculated values of the model after some internal changes."""
        if len(self._numericals) == 0:
            self._detS = xr.DataArray([])
            self._SInv = xr.DataArray([])
            self._S = xr.DataArray([])
            self._mu = xr.DataArray([])
        else:
            S = self._S

            invS = inv(S.values)
            self._SInv = xr.DataArray(data=invS, coords=S.coords, dims=S.dims)  # reuse coords from Sigma

            detS = abs(det(S.values)) ** -0.5
            if len(self._categoricals) == 0:
                self._detS = xr.DataArray(data=detS)  # no coordinates left to use...
            else:
                self._detS = xr.DataArray(data=detS, coords=self._p.coords, dims=self._p.dims)   # reuse coords from p

        if len(self._categoricals) == 0:
            self._p = xr.DataArray([])

        self._assert_no_nans()

        return self

    # #@profile
    # def _conditionout_continuous_internal_slow(self, cond_values, i, j, cat_keep, all_num_removing):
    #     if all_num_removing:
    #         detS_cond = 1  # TODO: uh.. is that right?
    #     # iterate the mu and sigma and p of each cg and update them
    #     #  for that create stacked _views_ on mu and sigma! it stacks up all categorical dimensions and thus
    #     #  allows us to iterate on them
    #     mu_stacked = self._mu.stack(pl_stack=cat_keep)
    #     S_stacked = self._S.stack(pl_stack=cat_keep)
    #     p_stacked = self._p.stack(pl_stack=cat_keep)
    #     detS_stacked = self._detS.stack(pl_stack=cat_keep)
    #     for coord in mu_stacked.pl_stack:
    #         indexer = dict(pl_stack=coord)
    #         mu = mu_stacked.loc[indexer]
    #         S = S_stacked.loc[indexer]
    #         detS = detS_stacked.loc[indexer]  # note: detS == abs(det(self._S))**-0.5 !!
    #
    #         diff_y_mu_J = cond_values - mu.loc[j]  # reused
    #         Sjj_inv = inv(S.loc[j, j])  # reused
    #
    #         # if all_num_removed:
    #         #    # detScond = 1  # this is set once above
    #         if not all_num_removing:
    #             # extent indexer to subselect only the part of mu and S that is updated. the rest is removed later.
    #             #  problem is: we cannot assign a shorter vector to stacked.loc[indexer]
    #             mu_indexer = dict(pl_stack=coord, mean=i)
    #             S_indexer = dict(pl_stack=coord, S1=i, S2=i)
    #
    #             # update Sigma and mu
    #             sigma_expr = np.dot(S.loc[i, j], Sjj_inv)  # reused below multiple times
    #             S_stacked.loc[S_indexer] = S.loc[i, i] - dot(sigma_expr, S.loc[j, i])  # upper Schur complement
    #             mu_stacked.loc[mu_indexer] = mu.loc[i] + dot(sigma_expr, diff_y_mu_J)
    #
    #             # for p update. Otherwise it is constant and calculated before the stacking loop
    #             detS_cond = abs(det(S_stacked.loc[S_indexer]))
    #
    #         # update p
    #         detQuotient = (detS_cond ** 0.5) * detS
    #         p_stacked.loc[indexer] *= detQuotient * exp(-0.5 * dot(diff_y_mu_J, dot(Sjj_inv, diff_y_mu_J)))

    def _name_idx_map (self, mode='num'):
        """Builds and returns a map of field names to their index within the order of fields of this mode.
        A normal user of this library never needs this function, as self.asindex provides the same functionality and should be preferred. However, internally in certain circumstances self.asindex cannot be used:

        (i) asindex is necessarily up-to-date in case there were both categorical and continuous variables marginalized out. It is only updated after marginalization of both, but not after the first one. This is because the abstract-model-level-update is called _after_the  whole condition-out operation.

        Args:
            mode: the mode. Currently only supports 'num'. A index map of all numerical fields will be returned.

        Internal:
            It relies on the correct coordinates of the 'mean' axis of self._mu.
        """
        if mode == 'num':
            # TODO: this it ugly. it relies on something too deep
            num_names = self._mu.coords['mean'].values.tolist()
            return {name: idx for idx, name in enumerate(num_names)}
        else:
            raise ValueError("invalid mode : ", str(mode))

    # @profile
    def _conditionout_continuous_internal_fast(self, p_, mu_, detS_, S_, cond_values, i_names, j_names, all_num_removing):
        if all_num_removing:
            detS_cond = 1

        # get numerical index for mu, sigma
        num_map = self._name_idx_map(mode='num')
        i = [num_map[v] for v in i_names]
        j = [num_map[v] for v in j_names]

        n = p_.size  # number of single gaussians in the cg
        m = len(self._numericals)  # gaussian dimension

        # extract numpy arrays and reshape to suitable form. This allows to iterate over the single parameters for S, mu, p by means of a standard python iterators
        p_np = p_.values.reshape(n)
        mu_np = mu_.values.reshape(n, m)
        detS_np = detS_.values.reshape(n)
        S_np = S_.values.reshape(n, m, m)

        for (idx, (p, mu, detS, S)) in enumerate(zip(p_np, mu_np, detS_np, S_np)):
            diff_y_mu_J = cond_values - mu[j]
            Sjj_inv = inv(S[ix_(j, j)])
            assert no_nan(Sjj_inv), "Inversion of Covariance Matrix failed."

            if not all_num_removing:
                # update Sigma and mu
                sigma_expr = np.dot(S[ix_(i, j)], Sjj_inv)  # reused below multiple times
                assert no_nan(sigma_expr), "Sigma_expr contains nan"
                S_np[idx][ix_(i, i)] -= dot(sigma_expr, S[ix_(j, i)])  # upper Schur complement
                mu_np[idx][i] += dot(sigma_expr, diff_y_mu_J)
                # this is for p update. Otherwise it is constant and calculated before the stacking loop
                detS_cond = abs(det(S_np[idx][ix_(i, i)]))

            # update p
            detQuotient = (detS_cond ** 0.5) * detS
            assert no_nan(detQuotient)
            p_np[idx] *= detQuotient * exp(-0.5 * dot(diff_y_mu_J, dot(Sjj_inv, diff_y_mu_J)))
            assert no_nan(p_np[idx])

    def _conditionout_continuous(self, num_remove):
        if len(num_remove) == 0:
            return

        # collect singular values to condition out
        cond_values = self._condition_values(num_remove)

        # TODO: this is ugly. We should incoorperate _numericals, _categoricals in the base model already, then we could put the normalization there as well

        if hasattr(self, 'opts') and self.opts['normalized']:  # this check for opts because cond_gaussians_wm don't have it, only mixable cgs and mcg use that function of cg wm
            cond_values = self._normalizer.norm(cond_values, mode="by name", names=num_remove)
            self._normalizer.update(num_remove=num_remove)

        # calculate updated mu and sigma for conditional distribution, according to GM script
        j = num_remove  # remove
        i = [name for name in self._numericals if name not in num_remove]  # keep

        cat_keep = self._mu.dims[:-1]
        all_num_removing = len(num_remove) == len(self._numericals)
        all_cat_removed = len(cat_keep) == 0

        if all_cat_removed:
            # special case: no categorical fields left. hence we cannot stack over them, it is only a single mu left
            # and we only need to update that
            sigma_expr = np.dot(self._S.loc[i, j], inv(self._S.loc[j, j]))  # reused below
            assert no_nan(sigma_expr), "Sigma_expr contains nan"
            self._S = self._S.loc[i, i] - dot(sigma_expr, self._S.loc[j, i])  # upper Schur complement
            self._mu = self._mu.loc[i] + dot(sigma_expr, cond_values - self._mu.loc[j])
            # update p: there is nothing to update. p is empty
        else:
            # this is the actual difficult case
            #self._conditionout_continuous_internal_slow(cond_values, i, j, cat_keep, all_num_removing)
            self._conditionout_continuous_internal_fast(self._p, self._mu, self._detS, self._S, cond_values, i, j, all_num_removing)

            # rescale to one
            # TODO: is this wrong? why do we not automatically get a normalized model?
            psum = self._p.sum()
            if psum != 0:
                self._p /= psum
            else:
                logger.warning("creating a conditional model with extremely low probability and alike low predictive "
                               "power")
                self._p.values = np.full_like(self._p.values, 1 / self._p.size)

            # in the conditionout_continuous_internal_* we partially updated only the relevant part of mu and Sigma
            # the remaining part is now removed, i.e. sliced out
            if all_num_removing:
                self._mu = xr.DataArray([])
                self._S = xr.DataArray([])
            else:
                self._mu = self._mu.loc[dict(mean=i)]
                self._S = self._S.loc[dict(S1=i, S2=i)]

        self._numericals = [name for name in self._numericals if name not in num_remove]

    def _conditionout_categorical(self, cat_remove):
        if len(cat_remove) == 0:
            return

        pairs = dict(self._condition_values(names=cat_remove, pairflag=True))

        # _p changes like in the categoricals.py case
        # trim the probability look-up table to the appropriate subrange and normalize it
        p = self._p.loc[pairs]
        self._p = p / p.sum()
        assert no_nan(self._p), "Renormalization of p failed."

        # _mu and _S is trimmed: keep the slice that we condition on, i.e. reuse the 'pairs' access-structure
        # note: if we condition on all categoricals this also works: it simply remains the single 'selected' mu...
        if len(self._numericals) != 0:
            self._mu = self._mu.loc[pairs]
            self._S = self._S.loc[pairs]

        # update internals
        self._categoricals = [name for name in self._categoricals if name not in cat_remove]

    def _conditionout(self, keep, remove):
        remove = set(remove)

        # condition on categorical fields
        cat_remove = [name for name in self._categoricals if name in remove]
        if len(cat_remove) > 0:
            self._conditionout_categorical(cat_remove)
            self._update()

        # condition on continuous fields
        num_remove = [name for name in self._numericals if name in remove]
        self._conditionout_continuous(num_remove)

        return self._unbound_updater,

    def _marginalizeout(self, keep, remove):
        # use weak marginals to get the best approximation of the marginal distribution that is still a cg-distribution
        keep = set(keep)
        num_keep = [name for name in self._numericals if name in keep]  # note: this is guaranteed to be sorted
        cat_remove = [name for name in self._categoricals if name not in keep]

        if len(self._categoricals) != 0:  # only enter if there is work to do
            # clone old p for later reuse
            if len(cat_remove) > 0:
                # marginalized p: just like in the categorical case (categoricals.py), i.e. sum over removed dimensions
                p = self._p.copy()
                self._p = self._p.sum(cat_remove)
            else:
                # no need to copy it
                p = self._p

        # marginalized mu and Sigma (taken from the script)
        if len(num_keep) != 0:
            # slice out the gaussian part to keep
            mu = self._mu.loc[dict(mean=num_keep)]
            S = self._S.loc[dict(S1=num_keep, S2=num_keep)]
            if len(cat_remove) == 0:
                # just set the sliced out gaussian parts
                self._mu = mu
                self._S = S
            else:
                # marginalized mu
                # sum over the categorical part to remove
                self._mu = (p * mu).sum(cat_remove) / self._p

                # marginalized Sigma - see script
                # only in that case the following operations yield something different from S
                mu_diff = mu - self._mu

                # outer product of each mu_diff.
                #  do it in numpy with einsum: 1st reshape to [x, len(mu)], 2nd use einsum
                #  credits to: http://stackoverflow.com/questions/20683725/numpy-multiple-outer-products
                shape = mu_diff.shape
                shape = (np.prod(shape[:-1]), shape[-1:][0])
                mu_diff_np = mu_diff.values.reshape(shape)
                mu_dyad = np.einsum('ij,ik->ijk' ,mu_diff_np, mu_diff_np)
                mu_dyad = mu_dyad.reshape(S.shape)  # match to shape of S

                inner_sum = mu_dyad + S

                times_p = inner_sum * p
                marginalized_sum = times_p.sum(cat_remove)
                normalized = marginalized_sum / self._p
                self._S = normalized

        # update fields and dependent variables
        self._categoricals = [name for name in self._categoricals if name in keep]
        self._numericals = num_keep
        return self._unbound_updater,

    def _density(self, x):
        cat_len = len(self._categoricals)
        num_len = len(self._numericals)
        cat = tuple(x[:cat_len])  # need it as a tuple for indexing below
        num = np.array(x[cat_len:])  # need as np array for dot product

        p = self._p.loc[cat].values

        if num_len == 0:
            return p

        # works because gaussian variables are - by design of this class - after categoricals.
        # Therefore the only not specified dimension is the last one, i.e. the one that holds the mean!
        mu = self._mu.loc[cat].values
        detS = self._detS.loc[cat].values
        invS = self._SInv.loc[cat].values
        xmu = num - mu
        gauss = (2 * pi) ** (-num_len / 2) * detS * exp(-.5 * np.dot(xmu, np.dot(invS, xmu)))
        assert no_nan(gauss), "Density computation failed."

        if cat_len == 0:
            return gauss
        else:
            return p * gauss

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        cat_len = len(self._categoricals)
        num_len = len(self._numericals)

        return _maximum_cgwm_heuristic1(cat_len, num_len, self._mu, self._p, self._detS)

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

