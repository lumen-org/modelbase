# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)

"""
@author: Philipp Lucas

This is a collection of methods to
  (i) fit parameters of various specific conditional gaussian model types to data, as well as
  (ii) conversion of a one parameter representation to another.

Data is generally given as a pandas data frame. Appropiate preperation is expected, e.g. normalization, no nans/nones, ...

Parameters are generally provided by means of numpy ndarrays. The order of categorical random variables in the given data frame is equivalent to the implicit order of dimensions (representing categorical random variables) in the derived parameters.

"""
from CGmodelselection.CG_CLZ_utils import CG_CLZ_Utils
from CGmodelselection.CG_MAP_utils import CG_MAP_Utils
from CGmodelselection.dataops import get_meta_data, prepare_cat_data

### model selection methods


def fit_pairwise_canonical(df):
    pass


def fit_clz_mean (df):
    """Fits parameters of a CLZ model to given data in pandas.DataFrame df and returns their representation as general mean paramters.

    Args:
        df: DataFrame of training data.
    """

    meta = get_meta_data(df)
    Y = df.as_matrix(meta['contnames'])
#    means, sigmas = standardizeContinuousData(Y) # required to avoid exp overflow
    D = prepare_cat_data(df[meta['catnames']], meta, method = 'dummy')  # transform discrete variables to indicator data
    # TODO: split into training and test data? if so: see Franks code
    solver = CG_CLZ_Utils(meta)  # initialize problem
    solver.drop_data(D, Y)  # set training data
    # solve it attribute .x contains the solution parameter vector.
    res = solver.solveSparse(klbda=0.2, verb=False, innercallback=solver.nocallback)
    #res = solver.solve(verb = 1, callback= solver.slimcallback) # without regularization - means far away
    clz_model_params = solver.getCanonicalParams(res.x, verb=False)

    (p, mus, Sigmas) = clz_model_params.getMeanParams(verb=False)
    return p, mus, Sigmas, meta


def fit_map_mean (df):
    """Fits parameters of a CLZ model to given data in pandas.DataFrame df and returns their representation as general mean paramters.

    Args:
        df: DataFrame of training data.
    """

    meta = get_meta_data(df)
    Y = df.as_matrix(meta['contnames'])
#    means, sigmas = standardizeContinuousData(Y) # required to avoid exp overflow
    D = prepare_cat_data(df[meta['catnames']], meta, method = 'flat')  # transform discrete variables to flat indices
    # TODO: split into training and test data? if so: see Franks code
    solver = CG_MAP_Utils(meta)  # initialize problem
    solver.drop_data(D, Y)  # set training data

    (p, mus, Sigmas) = solver.fitCG_variableCov(verb=False)
#    (p, mus, Sigmas) = solver.fitCG_fixedCov(verb = True)
    return p, mus, Sigmas, meta

### parameter transformation methods


def clz_to_mean ():
    """Transforms a set of parameters for a CLZ model (as generated by fit_clz_canonical()) to the equivalent general
    mean parameters."""
    pass


def mean_to_canonical ():
    """Transforms a set of general mean parameters to their equivalent general canonical parameter representation."""
    pass


def canonical_to_mean ():
    """Transforms a set of general canonical parameters to their equivalent general mean parameter representation."""
    pass
