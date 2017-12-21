# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import functools
import logging
import math
import numpy as np
from numpy import nan, pi, exp, dot, abs
from numpy.linalg import inv, det
from math import isnan
import xarray as xr
from scipy.optimize import minimize

import mb_modelbase.utils

from mb_modelbase.models_core import domains as dm
from mb_modelbase.utils import no_nan, validate_opts
from mb_modelbase.models_core import models as md
from mb_modelbase.models_core import cond_gaussian_wm as cgwm
from mb_modelbase.utils import utils

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def normalized(func):
    @functools.wraps(func)
    def wrapper(self, x):
        return self.denormalize(func(self.normalize(x)))
    return wrapper

def _argmax(p):
    return p.where(p == p.max(), drop=True)


def _masked(vec, mask):
    """Returns a vector of those entries vec[i] where mask[i] is True. Order of elements is invariant."""
    return [vec[i] for i in range(len(vec)) if mask[i]]


def _not_masked(vec, mask):
    """Returns a vector of those entries vec[i] where mask[i] is False. Order of elements is invariant."""
    return [vec[i] for i in range(len(vec)) if not mask[i]]


def _maximum_mixable_cg_heuristic_a(cat_len, marg_len, num_len, marginalized_mask, mu, p, detS):
    """ Returns an approximation to the point of maximum density.

    Heuristic (a) "highest single gaussian": return the density at the mean of the individually most probably gaussian (
     including shadowed ones). "individually" means that only the density of each individual mean of a gaussian
     is of interest, not the the actual, summed density at its mean (summed over shadowed gaussians)

    * this should be the simplest to implement, but also the worst performing?
    * in fact, this is cond_gaussian_wm._maximum_cgwm_heuristic1, but with the shadowed coordinates cut off
    """

    # get coordinates of maximum over all cgs including shadowed ones
    coord = cgwm._maximum_cgwm_heuristic1(cat_len + marg_len, num_len, mu, p, detS)
    # all_cat_len = len(self._categoricals) + len(self._marginalized)
    # num_len = len(self._numericals)

    # cut out shadowed fields
    return _not_masked(coord, marginalized_mask)


def _fix_args(fct, *args):
    """Fixes all but the first positional parameters and returns a function with these parameters fixed.
    e.g given fct = function of a, b, c it returns a
    """
    return lambda x: fct(x,*args)


def _density_mixture_cg(num, mu_, invS_, detS_, p_):
    """ Returns the density at num of the mixture of gaussian given by numpy arrays (not xarray)."""
    assert(len(num)>  0)
    n = p_.size  # number of single gaussians in the cg
    m = len(num)  # gaussian dimension
    prefactor = (2 * pi) ** (-m / 2)  # needed several times below

    # iterate over all at the same time. The reshape is necessary for use of the default iterator.
    gauss_sum = 0
    for p, mu, invS, detS in zip(p_.reshape(n), mu_.reshape(n, m), invS_.reshape(n, m, m), detS_.reshape(n)):
        xmu = num - mu
        gauss = prefactor * detS * exp(-.5 * np.dot(xmu, np.dot(invS, xmu)))
        assert (no_nan(gauss))
        gauss_sum += p * gauss

    assert (no_nan(gauss_sum))
    return gauss_sum


def _gradient_mixture_cg(x, mu_, invS_, detS_, p_):
    assert (len(x) > 0)
    n = p_.size  # number of single gaussians in the cg
    m = len(x)  # gaussian dimension
    prefactor = (2 * pi) ** (-m / 2)  # needed several times below

    gradient_sum = 0
    for p, mu, invS, detS in zip(p_.reshape(n), mu_.reshape(n, m), invS_.reshape(n, m, m), detS_.reshape(n)):
        xmu = x - mu
        common_sub1 = np.dot(invS, xmu)
        result = (prefactor * (detS ** -.5) * exp(-.5 * np.dot(xmu, common_sub1))) * (-.5 * common_sub1)
        gradient_sum += p * np.array(result).T[0]

    assert (no_nan(gradient_sum))
    return gradient_sum


def _maximum_cg(mus, Sinvs, Sdets, ps, num_len):
    """Returns an approximation to the point of maximum of density function and its value as a tuple (argmax, max)."""

    # make functions to calculate density and gradient of mixture
    p_fct = _fix_args(_density_mixture_cg, mus, Sinvs, Sdets, ps)
    dp_fct = _fix_args(_gradient_mixture_cg, mus, Sinvs, Sdets, ps)

    # get maximum over conditional
    mixture_max = None
    mixture_maxp = math.inf

    for mu in mus.reshape(-1, num_len):
        # TODO: fix newton cg: problem: doesn't work for non-scalar optimization problems
        #cur_density = minimize(lambda x: -p_fct(x), mu, method='Newton-CG', jac=lambda x: -dp_fct(x), tol=1e-6)
        cur_density = minimize(lambda x: -p_fct(x), mu, method='Nelder-Mead', tol=1e-6)
        cur_value = cur_density.x
        cur_density = cur_density.fun
        if cur_density < mixture_maxp:
            mixture_maxp = cur_density
            mixture_max = cur_value

    return (mixture_max, -mixture_maxp)


class Normalizer():
    """A normalizer provides methods to (de-) normalize a data vector according to a provided set of zscore normalization parameters.

    It is meant to be used as a plug-in component to a mixable conditional gaussian model, in the case that model has been learned on normalized data but is intended to be used as a model of the unnormalized data. Note that usually zscore normalization is applied in order to avoid numerical issues, and not for semantical reasons.

    See also the 'normalized_models.ipynb' in the notebook documentation directory.
    """

    def __init__(self, model, mean, stddev):
        self._model = model
        self._mean = np.array(mean, copy=True)
        self._stddev = np.array(stddev, copy=True)

        # index lookup within numericals only
        cat_len = len(model._categoricals)
        nums = model._numericals
        self._numname2idx = {name: model.asindex(name) - cat_len for name in nums}
        self._nums = list(nums)

    def update(self, num_keep=None, num_remove=None):
        """Updates the normalization information if fields are removed from the model. Give either the fields to keep
        or fields to keep."""
        if num_keep is not None and num_remove is not None:
            raise ValueError("You may not set both arguments, num_keep and num_remove!")
        if num_remove is not None:
            num_keep = utils.invert_sequence(num_remove, self._nums)

        if len(num_keep) == 0:
            self._stddev = self._mean = np.array([])
            self._numname2idx = {}
            self._nums = []
        else:
            num_keep = utils.sort_filter_list(num_keep, self._nums)  # bring num_keep in correct order
            n2i = self._numname2idx
            idxs = [n2i[n] for n in num_keep]
            self._stddev = self._stddev[idxs]
            self._mean = self._mean[idxs]

            self._numname2idx = {name: n2i[name] for name in num_keep}  # rebuild with remaining numerical field names
            self._nums = num_keep
        return self

    def norm(self, x, mode='all nums in order', **kwargs):
        if mode == 'all nums in order':
            assert (len(x) == len(self._nums))
            return (x - self._mean) / self._stddev
        if mode == 'all in order':
            raise NotImplemented("I am not sure if this works: can I rely on self._model._categoricals?")
            assert (len(x) == self._model.dim)
            cat_len = len(self._model._categoricals)
            x[cat_len:] = (x[cat_len:] - self._mean) / self._stddev
            return x
        elif mode == 'by name':
            names = kwargs['names']
            assert (len(x) == len(names))
            assert (len(x) <= len(self._nums))
            idxs = [self._numname2idx[n] for n in names]
            return (x - self._mean[idxs]) / self._stddev[idxs]
        else:
            raise ValueError("invalid mode")

    def denormalize (self, x, num_only=False):
        if not num_only:
            cat_len = len(self._model._categoricals)
            x[cat_len:] = x[cat_len:] * self._stddev + self._mean
            return x
        else:
            assert (len(x) == len(self._model._numericals))
            return x * self._stddev + self._mean

    def copy(self, model=None):
        model = self._model if model is None else model
        return Normalizer(model, self._mean, self._stddev)


class MixableCondGaussianModel(md.Model):
    """This is Conditional Gaussian (CG) model that supports true marginals and conditional.

     An advantage is, that this model stays more accurate when marginal and or conditional models are derived,
     compared to CG model with weak marginals (WM). See cond_gaussian_wm.py for more.

    A disadvantage is that is is much more computationally expensive to answer queries against this type of model.
    The reason is:
    (1) that marginals of mixable CG models are mixtures of CG models, and the query has to be answered for each
    component of the mixture.
    (2) the number of components grows exponentially with the number of discrete random variables (RV) marginalized.

    On the upside, however, the model complexity never becomes larger. In fact, it simply doesn't decrease if discrete
    RV are maginalized.

    It seems like this should be relatively straight forward. See the notes on paper.
    In essence we never throw parameters when marginalizing discrete RV, but just mark them as 'marginalized'.

    The major questions appear to be:

      * how to efficiently store what the marginalized variables are? what about 'outer interfaces' that rely on internal storage order or something? not sure if that really is a problem
      * how to determine the maximum of such a mixture? this is anyway an interesting question...!
      * can I reuse code from cond_gaussian_wm.py? I sure can, but I would love to avoid code duplication...

    """

    _fit_opts_allowed = {
        'fit_algo': set(['clz', 'map', 'full']),
        'normalized': set([True, False]),
    }

    def __init__(self, name):
        super().__init__(name)
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        self._categoricals = []  # list of categorical field names, which are not marginalized/conditioned out
        self._numericals = []
        # variables to deal with marginalization
        self._marginalized = []  # list of marginalized, discrete field names, NOT in same order like occurring in parameters
        self._marginalized_mask = xr.DataArray([])  # boolean xarray for all cat. fields (incl marginalized ones), where True indicates a marginalized field. coords are names of fields.
        self._all_categoricals = []  # list of both categorical field names: those remaining in the model and those marginalized out
        # parameters of the model
        self._p = xr.DataArray([])
        self._mu = xr.DataArray([])
        self._S = xr.DataArray([])
        # precomputed values to speed up model queries
        self._SInv = xr.DataArray([])
        self._detS = xr.DataArray([])
        # creates an self contained update function. we use it as a callback function later
        self._unbound_updater = functools.partial(self.__class__._update, self)

        # default options for this model. will be overwritten later with actual values.
        self.opts = {
            'fit_algo': 'full',
            'normalized': True,
        }
        self._normalizer = None

    def params2str(self):
        return (
            "p = \n" + str(self._p) + "\n" +
            "mu = \n" + str(self._mu) + "\n" +
            "S = \n" + str(self._S))

    def __str__(self):
        return self.params2str()

    def _set_model_params(self, p, mu, S, cats, nums):
        """Sets the model parameters to given values. """
        if self.opts['normalized']:
            logger.warning("cannot use normalized model with explicit model parameters. I disable the normalization.")
            self.opts['normalized'] = False

        fields = []
        # categorical fields can be derived
        coords = p.coords
        for cat in cats:
            extent = coords[cat].values.tolist()
            field = md.Field(cat, dm.DiscreteDomain(), dm.DiscreteDomain(extent), dtype='string')
            fields.append(field)

        # set numerical fields of abstract model class
        num_extents = cgwm.numeric_extents_from_params(S, mu, nums)
        for extent, num in zip(num_extents, nums):
            field = md.Field(num, dm.NumericDomain(), dm.NumericDomain(extent), dtype='numerical')
            fields.append(field)

        self.fields = fields

        # set model params
        self._p = p
        self._mu = mu
        self._S = S

        self._categoricals = cats
        self._numericals = nums

        self._marginalized_mask = xr.DataArray(data=[False] * len(self._categoricals), dims='name', coords=[self._categoricals])

        return self._unbound_updater,

    def _set_data(self, df, drop_silently):
        self._set_data_mixed(df, drop_silently)
        self._marginalized_mask = xr.DataArray(data=[False]*len(self._categoricals), dims='name', coords=[self._categoricals])
        return ()

    def _fit(self, **kwargs):
        assert (self.mode != 'none')
        validate_opts(kwargs, __class__._fit_opts_allowed)
        self.opts.update(kwargs)

        if self.opts['normalized']:
            df_norm, data_mean, data_stddev = md.normalize_dataframe(self.data, self._numericals)
            self._normalizer = Normalizer(self, data_mean, data_stddev)
        else:
            df_norm = self.data

        # choose fitting algo
        fit_algo = self.opts['fit_algo']
        if fit_algo == 'full':
            self._p, self._mu, self._S = cgwm.fit_full(df_norm, self.fields, self._categoricals, self._numericals)
        elif fit_algo == 'clz':
            self._p, self._mu, self._S = cgwm.fit_CLZ(df_norm, self._categoricals, self._numericals)
        elif fit_algo == 'map':
            self._p, self._mu, self._S = cgwm.fit_MAP(df_norm, self._categoricals, self._numericals)
        else:
            raise ValueError("invalid value for fit_algo: ", str(fit_algo))

        return self._unbound_updater,

    def _assert_invariants(self):
        """Check for some invariants and raise AssertionError if violated, since this represents a severe bug."""
        marg_set = set(self._marginalized)
        cat_set =  set(self._categoricals)
        p_set = set(self._p.dims) if len(self._p) > 0 else set()
        s_set = set(self._S.dims)
        invs_set = set(self._SInv.dims)
        dets_set = set(self._detS.dims)
        mu_set = set(self._mu.dims)

        # (1) non of the shadowed fields may still be in _categorical, i.e. the intersection is empty
        assert(len(marg_set & cat_set) == 0)
        # (2) all of the shadowed fields must exist in _p.dims
        assert(p_set.issuperset(marg_set))
        # (3) the union of shadowed field names and remaining categorical field names must be identical to _p.dims
        assert(marg_set | cat_set == p_set)
        # (4) dimensions of _S, _invS must be identical
        assert(s_set == invs_set)
        # (5) dimensions of _S[:-2] and _mu[:-1] and _p and dets_set must be identical
        if len(self._numericals) > 0:
            mu_set.remove("mean")
            s_set.remove("S1")
            s_set.remove("S2")
            assert(p_set == s_set == mu_set == dets_set)
        # (6) matching "True"s
        assert sum(self._marginalized_mask) == len(marg_set)
        # (7) marginalized mask must have proper length
        assert len(self._marginalized_mask) >= len(p_set)
        # (8) normalization mask must have proper length
        if self.opts['normalized']:
            assert len(self._normalizer._nums) == len(self._numericals)
            assert list(self._normalizer._nums) == self._numericals
        # (9) assert there is no nans in parameters
        self._assert_no_nans()

    def _update(self):   # mostly like CG WM, but different update for _detS
        """Updates dependent parameters / precalculated values of the model after some internal changes."""
        if len(self._numericals) == 0:
            self._S = xr.DataArray([])
            self._SInv = xr.DataArray([])
            self._detS = xr.DataArray([])
            self._mu = xr.DataArray([])
        else:
            S = self._S

            invS = inv(S.values)
            self._SInv = xr.DataArray(data=invS, coords=S.coords, dims=S.dims)  # reuse coords from Sigma

            detS = abs(det(S.values)) ** -0.5
            if len(self._categoricals) == 0 and len(self._marginalized) == 0:
                self._detS = xr.DataArray(data=detS)  # no coordinates left to use... none needed, its a scalar now!
            else:
                self._detS = xr.DataArray(data=detS, coords=self._p.coords, dims=self._p.dims)   # reuse coords from p

        if len(self._categoricals) == 0 and len(self._marginalized) == 0:
            self._p = xr.DataArray([])

        self._assert_invariants()

        return self

    def _update_marginalized(self, cat_marginalized=None, cat_conditioned=None):
        """Updates the marginalized information after some categorical dimensions have been conditioned out.
        Args:
            Specify either the categoricals kept in the model, or those removed.
        """
        mask = self._marginalized_mask

        if cat_marginalized is not None:
            self._marginalized += cat_marginalized
            for name in cat_marginalized:
                assert (not mask.loc[name])
                mask.loc[name] = True

        if cat_conditioned is not None:
            cat_keep = utils.invert_sequence(cat_conditioned, mask.coords['name'].values.tolist())
            self._marginalized_mask = mask.loc[cat_keep]
            assert sum(self._marginalized_mask) == len(self._marginalized)

    def _marginalizeout(self, keep, remove):
        # we use exact marginals, which lead to a kind of mixtures of cg. Alternatively speaking: we simply keep the
        # full parameters but shadow the existence of the marginalized field to the outside.
        keep = set(keep)
        num_keep = [name for name in self._numericals if name in keep]  # note: this is guaranteed to be sorted
        cat_remove = [name for name in self._categoricals if name not in keep]

        # we never actually marginalize out categorical fields. we simply shadow them, but keep parameters internally

        # marginalized mu and Sigma (taken from the script)
        if len(num_keep) != 0:
            # marginalize numerical fields: slice out the gaussian part to keep
            self._mu = self._mu.loc[dict(mean=num_keep)]
            self._S = self._S.loc[dict(S1=num_keep, S2=num_keep)]

        if self.opts['normalized']:
            self._normalizer.update(num_keep=num_keep)

        # update fields and dependent variables
        self._categoricals = [name for name in self._categoricals if name in keep]
        self._numericals = num_keep

        # mark newly marginalized categorical fields as such
        self._update_marginalized(cat_marginalized=cat_remove)

        return self._unbound_updater,

    # reuse methods of non-mixed cgs
    _conditionout_continuous = cgwm.CgWmModel._conditionout_continuous
    _conditionout_categorical = cgwm.CgWmModel._conditionout_categorical
    _conditionout_continuous_internal_fast = cgwm.CgWmModel._conditionout_continuous_internal_fast
    #_conditionout_continuous_internal_slow = cgwm.CgWmModel._conditionout_continuous_internal_slow

    # reuse of internal utility methods
    _assert_no_nans = cgwm.CgWmModel._assert_no_nans  # used in several places
    _name_idx_map = cgwm.CgWmModel._name_idx_map  # used in _conditionout_continuous_internal_fast

    def _conditionout(self, keep, remove):
        remove = set(remove)

        # condition on categorical fields
        cat_remove = [name for name in self._categoricals if name in remove]
        if len(cat_remove) > 0:
            self._conditionout_categorical(cat_remove)
            self._update_marginalized(cat_conditioned=cat_remove)
            self._update()

        # condition on continuous fields
        num_remove = [name for name in self._numericals if name in remove]

        self._conditionout_continuous(num_remove)

        return self._unbound_updater,

    def _density_internal_stacked(self, num, mu_shadowed, invS_shadowed, detS_shadowed, p_shadowed):
        ## alternative 2 (see above)
        # stack over shadowed dimensions

        num_len = len(self._numericals)
        mu_stacked = mu_shadowed.stack(pl_stack=self._marginalized)
        invS_stacked = invS_shadowed.stack(pl_stack=self._marginalized)
        detS_stacked = detS_shadowed.stack(pl_stack=self._marginalized)
        p_stacked = p_shadowed.stack(pl_stack=self._marginalized)
        gauss_sum = 0
        prefactor = (2 * pi) ** (-num_len / 2)  # needed several times below
        for coord in mu_stacked.pl_stack:
            indexer = dict(pl_stack=coord)
            mu = mu_stacked.loc[indexer].values
            invS = invS_stacked.loc[indexer].values
            detS = detS_stacked.loc[indexer].values
            p = p_stacked.loc[indexer].values
            xmu = num - mu
            gauss = prefactor * detS * exp(-.5 * np.dot(xmu, np.dot(invS, xmu)))
            gauss_sum += p * gauss

        return gauss_sum

    # @profile
    def _density(self, x):
        """Returns the density of the model at x.

        Internal:

        * all shadowed fields are categorical
        * categorical fields are needed to retrieve the correct mu and sigma to then query the density for that gaussian
        * instead of one gaussian, we have to compute the density at x for all implicit gaussians (by shadowed categorical
         fields) and multiply it with the respective probability of occurrence (self._p)
        * then, these weighted densities have to be summed

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
           * this is a bad idea, because we cross join is already there - I simply need to iterate over it -> see alternative 2

        * alternative 2:
           - filter mu and sigma by the given categorical values in x,
           - that results in all the gaussians that we need to evaluate on the numerical part of x
           - stack over filtered mu and sigma (like in _conditionout of cg wm), to iterate over it
           - then sum densities up
        """
        cat_len = len(self._categoricals)
        num_len = len(self._numericals)
        num = np.array(x[cat_len:])  # need as np array for dot product
        cat = tuple(x[:cat_len])   # need it as a tuple for indexing

        if self.opts['normalized']:
            num = self._normalizer.norm(num)

        if num_len == 0:
            if len(self._marginalized) == 0:
                p = self._p
            else:
                p = self._p.sum(self._marginalized)   # sum over marginalized/shadowed fields
            return p.loc[cat].values

        if len(self._marginalized) == 0:
            # no shadowed/marginalized fields. hence we cannot stack over them and the density query works as "normal"
            # the following is copy-and-pasted from cond_gaussian_wm.py -> _density
            mu = self._mu.loc[cat].values
            detS = self._detS.loc[cat].values
            invS = self._SInv.loc[cat].values
            xmu = num - mu
            gauss = (2 * pi) ** (-num_len / 2) * detS * exp(-.5 * np.dot(xmu, np.dot(invS, xmu)))

            if cat_len == 0:
                result = gauss
            else:
                p = self._p.loc[cat].values
                result = p * gauss
        else:
            # dictionary of "categorical-field-name : value" for all given categorical values

            # note: even if cat_len == 0, i.e. the following selections have 'empty cat_dict', it works
            cat_dict = dict(zip(self._categoricals, cat))

            # filter mu and sigma by the given categorical values in x, i.e. a view on these xarrays
            mu_shadowed = self._mu.loc[cat_dict]
            invS_shadowed = self._SInv.loc[cat_dict]
            detS_shadowed = self._detS.loc[cat_dict]
            p_shadowed = self._p.loc[cat_dict]

            result = _density_mixture_cg(num, mu_shadowed.values, invS_shadowed.values, detS_shadowed.values, p_shadowed.values)
            #return self._density_internal_stacked(num, mu_shadowed, invS_shadowed, detS_shadowed, p_shadowed)

            ## vectorized version (instead of alternative 2 code)
            # I simply don't know the syntax for what i want...
            # maybe it is not possible?
            # np.dot(xmu, np.dot(invS, xmu))
            # .sum()*(2 * pi) ** (-num_len / 2)

        return result

    def _maximum_mixable_cg_heuristic_b(self):
        """ Returns an approximation to the point of maximum density.

        Heuristic "highest accumulated gaussian":
         1. calculate cumulated density at each gaussian (including shadowed ones)
         2. take the maximum of these, but cut off shadowed coordinates
        This is probably the best performing one of the maximum heuristics for this model type.
        """

        # stack over all means (including the shadowed ones) in self_mu and keep coordinates of maximum cumulated density
        stack_over = self._marginalized + self._categoricals
        len_so = len(stack_over)
        mu_stacked = self._mu.stack(pl_stack=stack_over)
        p_max = -np.inf
        coord_max = None
        for mu_coord in mu_stacked.pl_stack:
            # assemble coordinates for density query
            mu_coord = mu_coord.item()
            mu_indexer = dict(pl_stack=mu_coord)
            num_coord = mu_stacked.loc[mu_indexer].values.tolist()
            mu_coord = (mu_coord,) if len_so == 1 else mu_coord  # make sure it's a tuple (needed for _not_masked())
            cat_coord = _not_masked(mu_coord, self._marginalized_mask)  # remove coordinates of marginalized fields
            coord = cat_coord + num_coord

            # query density and compare to so-far maximum
            # TODO: can I use pseudo-density like in cond_gaussian_wm._maximum_cgwm_heuristic1?
            p = self._density(coord)
            assert (no_nan(p))
            if p > p_max:
                p_max = p
                coord_max = coord

        assert(coord_max is not None)
        return coord_max

    def _maximum_mixable_cg_heuristic_d(self):
        num_len = len(self._numericals)

        if len(self._categoricals) == 0:
            # only a single mixture left ... Note: index 0 is for the arg max (and not the max)
            return _maximum_cg(self._mu.values, self._SInv.values, self._detS.values, self._p.values, num_len)[0]

        mixable_maximum_cat = None
        mixable_maximum_num = None
        mixable_maximum_density = -math.inf

        # iterate over conditionals
        mu_stacked = self._mu.stack(pl_stack=self._categoricals)
        invS_stacked = self._SInv.stack(pl_stack=self._categoricals)
        detS_stacked = self._detS.stack(pl_stack=self._categoricals)
        p_stacked = self._p.stack(pl_stack=self._categoricals)

        for coord in mu_stacked.pl_stack:
            # extract paramters of mixture of gaussian
            indexer = dict(pl_stack=coord)
            mus = mu_stacked.loc[indexer].values
            Sinvs = invS_stacked.loc[indexer].values
            Sdets = detS_stacked.loc[indexer].values
            ps = p_stacked.loc[indexer].values

            mixture_max, mixture_maxp = _maximum_cg(mus, Sinvs, Sdets, ps, num_len)

            if mixable_maximum_density < mixture_maxp:
                mixable_maximum_density = mixture_maxp
                mixable_maximum_num = mixture_max
                mixable_maximum_cat = coord.item()

        return list(mixable_maximum_cat) + mixable_maximum_num.tolist()

    def _maximum(self):

        cat_len = len(self._categoricals)
        num_len = len(self._numericals)
        mrg_len = len(self._marginalized)

        if mrg_len == 0:
            # then we do not have a conditional mixture of gaussian distribution, but a normal conditional gaussian dist
            result = cgwm._maximum_cgwm_heuristic1(cat_len, num_len, self._mu, self._p, self._detS)

        elif num_len == 0:
            # find maximum in p and return its coordinates
            p = self._p.sum(self._marginalized)  # sum over marginalized fields
            pmax = _argmax(p)  # get view on maximum (coordinates remain)
            result = [idx[0] for idx in pmax.indexes.values()]  # extract coordinates from indexes

        else:
            # this is the difficult case, and we don't have a perfect solution yet, just a couple of heuristics...
            # return _maximum_mixable_cg_heuristic_a( cat_len, mrg_len, num_len, self._marginalized_mask, self._mu, self._p, self._detS)
            #result = self._maximum_mixable_cg_heuristic_b()
            result = self._maximum_mixable_cg_heuristic_d()

        assert(result is not None)
        return self._normalizer.denormalize(result) if self.opts['normalized'] else result

    # mostly like cg wm
    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._mu = self._mu.copy()
        mycopy._S = self._S.copy()
        mycopy._p = self._p.copy()
        mycopy._categoricals = self._categoricals.copy()
        mycopy._numericals = self._numericals.copy()
        mycopy._marginalized = self._marginalized.copy()
        mycopy._marginalized_mask = self._marginalized_mask.copy()
        mycopy.opts = self.opts.copy()
        if self.opts['normalized']:
            mycopy._normalizer = self._normalizer.copy(mycopy)
        mycopy._update()
        return mycopy


if __name__ == '__main__':

    # load data
    data = crabs.mixed('data/crabs/australian-crabs.csv')

    # fit model
    m = MixableCondGaussianModel(name="foo_model")
    m.fit(df=data)

    orig = m.copy()

    m = m.model(model=["RW"])
    aggr = m.aggregate("maximum")

    print(aggr)
