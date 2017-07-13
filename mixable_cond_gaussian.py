# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import functools
import logging
import numpy as np
from numpy import nan, pi, exp, dot, abs
from numpy.linalg import inv, det
import xarray as xr

import models as md
import cond_gaussian_wm as cg

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MixableCondGaussianModel(md.Model):
    """This is Conditional Gaussian (CG) model that supports true marginals and conditional.

     An advantage is, that this model stays more accurate when marginal and or conditional models are derived, compared to CG model with weak marginals (WM). See cond_gaussian_wm.py for more.

    A disadvantage is that is is much more computationally expensive to answer queries against this type of model. The reason is:
     (1) that marginals of mixable CG models are mixtures of CG models, and the query has to be answered for each component of the mixture.
     (2) the number of components grows exponentially with the number of discrete random variables (RV) marginalized.

    It seems like this should be relatively straight forward. See the notes on paper.
    In essence we never throw parameters when marginalizing discrete RV, but just mark them as 'marginalized'.

    The major questions appear to be:

      * how to efficiently store what the marginalized variables are? what about 'outer interfaces' that rely on internal storage order or something? not sure if that really is a problem
      * how to determine the maximum of such a mixture? this is anyway an interesting question...!
      * can I reuse code from cond_gaussian_wm.py? I sure can, but I would love to avoid code duplication...

    """
    pass

    def __init__(self, name):
        super().__init__(name)

        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        self._marginalized = []  # list of marginalized, discrete fields
        self._categoricals = []
        self._numericals = []
        self._p = xr.DataArray([])
        self._mu = xr.DataArray([])
        self._S = xr.DataArray([])
        self._SInv = xr.DataArray([])
        self._detS = xr.DataArray([])
        # creates an self contained update function. we use it as a callback function later
        self._unbound_updater = functools.partial(self.__class__._update, self)

    def _set_data(self, df, drop_silently):    # exactly like CG WM!!
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        assert (self.mode != 'none')
        self._p, self._mu, self._S = cg.fitConditionalGaussian(self.data, self.fields, self._categoricals,
                                                            self._numericals)
        return self._unbound_updater,

    def _update(self):   # exactly like CG WM!!
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

        return self

    def _marginalizeout(self, keep, remove):
        # we use exact marginals, which lead to a kind of mixtures of cg. Alternatively speaking: we simply keep the
        # full parameters but shadow the existence of the marginalized field to the outside.
        keep = set(keep)
        num_keep = [name for name in self._numericals if name in keep]  # note: this is guaranteed to be sorted
        cat_remove = [name for name in self._categoricals if name not in keep]

        # we never actually marginalize out categorical fields. we simply shadow them, but keep the parameters
        # internally
        # if len(self._categoricals) != 0:  # only enter if there is work to do
        #     # clone old p for later reuse
        #     if len(cat_remove) > 0:
        #         # marginalized p: just like in the categorical case (categoricals.py), i.e. sum over removed dimensions
        #         p = self._p.copy()
        #         self._p = self._p.sum(cat_remove)
        #     else:
        #         # no need to copy it
        #         p = self._p

        # marginalized mu and Sigma (taken from the script)
        if len(num_keep) != 0:
            # marginalize numerical fields: slice out the gaussian part to keep
            self._mu = self._mu.loc[dict(mean=num_keep)]
            self._S = self._S.loc[dict(S1=num_keep, S2=num_keep)]
            # mu = self._mu.loc[dict(mean=num_keep)]
            # S = self._S.loc[dict(S1=num_keep, S2=num_keep)]

            # mark categorial fields to marginalize as such
            self._marginalized += cat_remove

            # # marginalize categorical fields:
            # if len(cat_remove) == 0:
            #     # just set the sliced out gaussian parts
            #     self._mu = mu
            #     self._S = S
            # else:
            #     # marginalized mu
            #     # sum over the categorical part to remove
            #     self._mu = (p * mu).sum(cat_remove) / self._p
            #
            #     # marginalized Sigma - see script
            #     # only in that case the following operations yield something different from S
            #     mu_diff = mu - self._mu
            #
            #     # outer product of each mu_diff.
            #     #  do it in numpy with einsum: 1st reshape to [x, len(mu)], 2nd use einsum
            #     #  credits to: http://stackoverflow.com/questions/20683725/numpy-multiple-outer-products
            #     shape = mu_diff.shape
            #     shape = (np.prod(shape[:-1]), shape[-1:][0])
            #     mu_diff_np = mu_diff.values.reshape(shape)
            #     mu_dyad = np.einsum('ij,ik->ijk' ,mu_diff_np, mu_diff_np)
            #     mu_dyad = mu_dyad.reshape(S.shape)  # match to shape of S
            #
            #     inner_sum = mu_dyad + S
            #
            #     times_p = inner_sum * p
            #     marginalized_sum = times_p.sum(cat_remove)
            #     normalized = marginalized_sum / self._p
            #     self._S = normalized

        # update fields and dependent variables
        self._categoricals = [name for name in self._categoricals if name in keep]
        self._numericals = num_keep
        return self._unbound_updater,

    def _conditionout(self, keep, remove):   # exactly like CG WM!!
        remove = set(remove)

        # condition on categorical fields
        cat_remove = [name for name in self._categoricals if name in remove]
        if len(cat_remove) != 0:
            pairs = dict(self._condition_values(cat_remove, True))

            # _p changes like in the categoricals.py case
            # trim the probability look-up table to the appropriate subrange and normalize it
            p = self._p.loc[pairs]
            self._p = p / p.sum()

            # _mu is trimmed: keep the slice that we condition on, i.e. reuse the 'pairs' access-structure
            # note: if we condition on all categoricals this also works: it simply remains the single 'selected' mu...
            if len(self._numericals) != 0:
                self._mu = self._mu.loc[pairs]
                self._S = self._S.loc[pairs]

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

            cat_keep = self._mu.dims[:-1]
            if len(cat_keep) != 0:
                # iterate the mu and sigma of each cg and update them
                #  for that create stacked _views_ on mu and sigma! it stacks up all categorical dimensions and thus
                #  allows us to iterate on them
                # TODO: can I use the same access structure or do i need seperate ones for mu and S?
                mu_stacked = self._mu.stack(pl_stack=cat_keep)
                S_stacked = self._S.stack(pl_stack=cat_keep)
                for mu_coord, S_coord in zip(mu_stacked.pl_stack, S_stacked.pl_stack):
                    mu_indexer = dict(pl_stack=mu_coord)
                    S_indexer = dict(pl_stack=S_coord)

                    mu = mu_stacked.loc[mu_indexer]
                    S = S_stacked.loc[S_indexer]

                    # extent indexer to subselect only the part of mu and S that is updated. the rest is removed later.
                    #  problem is: we cannot assign a shorter vector to stacked.loc[indexer]
                    mu_indexer['mean'] = i
                    S_indexer['S1'] = i
                    S_indexer['S2'] = i

                    # update Sigma and mu
                    sigma_expr = np.dot(S.loc[i, j], inv(S.loc[j, j]))   # reused below multiple times
                    S_stacked.loc[S_indexer] = S.loc[i, i] - dot(sigma_expr, S.loc[j, i])  # upper Schur complement
                    mu_stacked.loc[mu_indexer] = mu.loc[i] + dot(sigma_expr, condvalues - mu.loc[j])

                # above we partially updated only the relevant part of mu and Sigma. the remaining part is now removed:
                self._mu = self._mu.loc[dict(mean=i)]
                self._S = self._S.loc[dict(S1=i,S2=i)]
            else:
                # special case: no categorical fields left. hence we cannot stack over them, it is only a single mu left
                # and we only need to update that
                sigma_expr = np.dot(self._S.loc[i, j], inv(self._S.loc[j, j]))  # reused below
                self._S = self._S.loc[i, i] - dot(sigma_expr, self._S.loc[j, i])  # upper Schur complement
                self._mu = self._mu.loc[i] + dot(sigma_expr, condvalues - self._mu.loc[j])

        # remove fields as needed
        self._categoricals = [name for name in self._categoricals if name not in remove]
        self._numericals = [name for name in self._numericals if name not in remove]
        return self._unbound_updater,

    def _density(self, x):
        """Returns the density of the model at x.

        Internal:

        * all shadowed fields are categorical
        * categorical fields are needed to retrieve the correct mu and sigma to then query the density for that gaussian
        * instead of one gaussian, we have to compute the density at x for all implicit gaussians (by shadowed categorical
         fields)
        * then, the densities have to summed over

        * question is: how can I do this in parallel, without explicitly iterating through all value combinations of
          the shadowed fields?

        * alternative iterative procedure:
           - create data frame of cross join of marginalized field values
           - init sum to 0
           - iterate over that frame:
              - extract the corresponding gaussian
              - compute density for gaussian
              - add to sum
           - return that sum

        * alternative 2:
           - filter mu and sigma by the given categorical values in x,
           - that results in all the gaussians that we need to evaluate on the numerical part of x
           - stack over filtered mu and sigma (like in _conditionout of cg wm), to iterate over it
           - then sum densities up
        """
        cat_len = len(self._categoricals)
        num_len = len(self._numericals)
        #cat = tuple(x[:cat_len])  # need it as a tuple for indexing below
        num = np.array(x[cat_len:])  # need as np array for dot product

        # dictionary of "categorical-field-name : value" for all given categorical values
        cat_dict = dict(zip(self._categoricals, x[:cat_len]))

        # filter mu and sigma by the given categorical values in x
        mu_shadowed = self._mu.loc[cat_dict]
        invS_shadowed = self._SInv.loc[cat_dict]
        detS_shadowed = self._detS.loc[cat_dict]

        ## alternative 2 (see above)
        # stack over shadowed dimensions
        mu_stacked = mu_shadowed.stack(pl_stack=self._marginalized)
        invS_stacked = invS_shadowed.stack(pl_stack=self._marginalized)
        detS_stacked = detS_shadowed.stack(pl_stack=self._marginalized)
        psum = 0
        for mu_coord, SInv_coord, detS_coord in zip(mu_stacked.pl_stack, invS_stacked.pl_stack, detS_stacked.pl_stack):
            mu_indexer = dict(pl_stack=mu_coord)
            SInv_indexer = dict(pl_stack=SInv_coord)
            detS_indexer = dict(pl_stack=detS_coord)

            mu = mu_stacked.loc[mu_indexer]
            invS = invS_stacked.loc[SInv_indexer]
            detS = detS_stacked.loc[detS_indexer]

            xmu = num - mu
            p = (2 * pi) ** (-num_len / 2) * detS * exp(-.5 * np.dot(xmu, np.dot(invS, xmu)))

            psum += p

        return psum

        ## vectorized version (instead of alternative 2 code)
        # I simply don't know the syntax for what i want...
        # np.dot(xmu, np.dot(invS, xmu))
        #.sum()*(2 * pi) ** (-num_len / 2)


#    def _sample(self):
#        pass

    def copy(self, name=None):
        pass


