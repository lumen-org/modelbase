# Copyright (c) 2016 Philipp Lucas and Frank Nussbaum, FSU Jena
import logging
import numpy as np
from numpy import ix_, nan, pi, exp, dot
from numpy.linalg import inv
import pandas as pd
import xarray as xr

import utils
import models as md
import domains as dm

#imports frank
from cond_gaussian.datasampling import genCGSample, genCatData, genCatDataJEx
from cond_gaussian.output import plothist

# setup logger
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
        self._S = np.array([])
        self._SInv = nan
        self._detS = nan

    @staticmethod
    def _fitFullLikelihood(data, fields, dc):
        """fit full likelihood for CG model. the data frame data consists of dc many categorical columns and the rest are
        numerical columns. all categorical columns occure before the numercial ones."""
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

            #        print(pMLx)

        it = np.nditer(pML, flags=['multi_index'])  # iterator over complete array
        while not it.finished:
            ind = it.multi_index
            #            print "%d <%s>" % (it[0], it.multi_index)
            #            print(ind, pMLx[ind])
            musML[ind] /= pML[ind]
            it.iternext()
        pML /= 1.0 * n

        Sigma = np.zeros((dg, dg))
        for row in data.itertuples():
            cats = row[1:1 + dc]
            gauss = row[1 + dc:]
            ymu = gauss - musML.loc[cats]
            Sigma += np.outer(ymu, ymu)

        Sigma /= n
        #Sigma = xr.DataArray(Sigma, coords=[gausscols]*2)

        return pML, musML, Sigma

    def fit(self, df):
        """Fits the model to passed DataFrame.

        Parameters:
            df: A pandas data frame that holds the data to fit the model to. All columns of df are used.

        Internal:
            This method estimates the set of mean parameters that fit best to the data given in the dataframe df.

        Returns:
            The modified model with selected parameters set.
        """
        # split in categorical and numeric columns
        categoricals = []
        numericals = []
        for colname in df:
            column = df[colname]
            # if column.dtype == "category" or column.dtype == "object":
            if column.dtype == "object":
                categoricals.append(colname)
            else:
                numericals.append(colname)

        # reorder data frame such that categorical columns are first
        df = pd.DataFrame(df, columns=categoricals + numericals)

        #  derive fields
        fields = []
        for colname in categoricals:
            column = df[colname]
            domain = dm.DiscreteDomain()
            extent = dm.DiscreteDomain(sorted(column.unique()))
            field = md.Field(colname, domain, extent, 'string')
            fields.append(field)
        for colname in numericals:
            column = df[colname]
            field = md.Field(colname, dm.NumericDomain(), dm.NumericDomain(column.min(), column.max()), 'numerical')
            fields.append(field)
        self.fields = fields
        self._update()

        dc = len(categoricals)

        (p, mus, Sigma) = ConditionallyGaussianModel._fitFullLikelihood(df, fields, dc)
        self._p = p
        self._mu = mus
        self._S = Sigma
        self._categoricals = categoricals
        self._numericals = numericals

        return self.update()

    @staticmethod
    def cg_dummy():
        """Returns a dataframe that contains sample of a 4d cg distribution. See the code for the used parameters."""

        # chose fixed parameters
        mu_M_Jena = [0, 0]
        mu_F_Jena = [1, 3]
        mu_M_Erfurt = [-10, 1]
        mu_F_Erfurt = [-5, -6]
        p_M_Jena = 0.35
        p_F_Jena = 0.25
        p_M_Erfurt = 0.1
        p_F_Erfurt = 0.3
        S = [[3, 0.5], [0.5, 1]]
        dims = ['sex', 'city', 'age', 'income']
        # and a sample size
        samplecnt = 1000

        # generate samples for each and arrange in dataframe
        df_cat = pd.concat([
            pd.DataFrame([["M", "Jena"]] * round(samplecnt * p_M_Jena), columns=['sex', 'city']),
            pd.DataFrame([["F", "Jena"]] * round(samplecnt * p_F_Jena), columns=['sex', 'city']),
            pd.DataFrame([["M", "Erfurt"]] * round(samplecnt * p_M_Erfurt), columns=['sex', 'city']),
            pd.DataFrame([["F", "Erfurt"]] * round(samplecnt * p_F_Erfurt), columns=['sex', 'city'])
        ])

        df_num = pd.concat([
            pd.DataFrame(np.random.multivariate_normal(mu_M_Jena, S, round(samplecnt * p_M_Jena)), columns=['age', 'income']),
            pd.DataFrame(np.random.multivariate_normal(mu_F_Jena, S, round(samplecnt * p_F_Jena)), columns=['age', 'income']),
            pd.DataFrame(np.random.multivariate_normal(mu_M_Erfurt, S, round(samplecnt * p_M_Erfurt)), columns=['age', 'income']),
            pd.DataFrame(np.random.multivariate_normal(mu_F_Erfurt, S, round(samplecnt * p_F_Erfurt)), columns=['age', 'income'])
        ])
        df = pd.concat([df_cat, df_num], axis=1)
#        df.plot.scatter(x="age", y="income")
        return df

    def update(self):
        """Updates dependent parameters / precalculated values of the model after some internal changes."""
        self._update()

        if len(self._numericals) == 0:
            self._detS = nan
            self._SInv = nan
            self._S = np.array([])
            self._mu = xr.DataArray([])
        else:
            self._detS = np.abs(np.linalg.det(self._S))
            self._SInv = np.linalg.inv(self._S)

        if len(self._categoricals) == 0:
            self._p = xr.DataArray([])

        return self

    def _conditionout(self, remove):
        if len(remove) == 0 or self._isempty():
            return self
        if len(remove) == self._n:
            return self._setempty()

        remove = set(remove)

        # condition on categorical fields
        # _S remains unchanged
        cat_remove = [self.byname(name) for name in self._categoricals if name in remove]
        if len(cat_remove) != 0:
            # note: if we condition on all categoricals the following procedure also works. It simply remains the single
            # 'selected' mu...
            # _p changes like in the categoricals.py case
            pairs = []
            # todo: factor this out. its the same as in categoricals and we can put it as a function there
            for field in cat_remove:
                domain = field['domain']
                dvalue = domain.value()
                assert (domain.isbounded())
                if field['dtype'] == 'string':
                    # TODO: we don't know yet how to condition on a not singular, but not unrestricted domain.
                    pairs.append((field['name'], dvalue if domain.issingular() else dvalue[0]))
                else:
                    raise ValueError('invalid dtype of field: ' + str(field['dtype']))

            # trim the probability look-up table to the appropriate subrange and normalize it
            p = self._p.loc[dict(pairs)]
            self._p = p / p.sum()

            # _mu is trimmed: keep the slice that we condition on, i.e. reuse the 'pairs' access-structure
            self._mu = self._mu.loc[dict(pairs)]

        # condition on continuous fields
        num_remove = [name for name in self._numericals if name in remove]  # guaranteed to be sorted!
        #if len(num_remove) == len(self._numericals):
        #    all gaussians are implicitely removed
        if len(num_remove) != 0:
            # collect singular values to condition out
            condvalues = []
            # todo: factor this out. its the same as in gaussians and we can put it as a function there
            for name in num_remove:
                field = self.byname(name)
                domain = field['domain']
                dvalue = domain.value()
                assert (domain.isbounded())
                if field['dtype'] == 'numerical':
                    condvalues.append(dvalue if domain.issingular() else (dvalue[1] - dvalue[0]) / 2)
                    # TODO: we don't know yet how to condition on a not singular, but not unrestricted domain.
                else:
                    raise ValueError('invalid dtype of field: ' + str(field['dtype']))
            #condvalues = matrix(condvalues).T

            # calculate updated mu and sigma for conditional distribution, according to GM script
            i = [idx - len(self._categoricals) for idx in self.asindex(num_remove)]
            j = utils.invert_indexes(i, len(self._numericals))

            #S = matrix(self._S)
            #Sigma_expr = S[ix_(i, j)] * S[ix_(j, j)].I  # needed for update of mu later
            S = self._S
            Sigma_expr = np.dot(S[ix_(i, j)], inv(S[ix_(j, j)]))
            self._S = S[ix_(i, i)] - dot(Sigma_expr, S[ix_(j, i)])  # upper Schur complement

            cat_keep = self._mu.dims[1:]
            if len(cat_keep) != 0:
                # iterate over all mu and update them
                stacked = self._mu.stack(pl_stack=cat_keep)  # this is a reference to mu!
                for coord in stacked.pl_stack:
                    indexer = dict(pl_stack=coord)
                    mu = stacked.loc[indexer]
                    # todo: can't i write: mu = ...  ?
                    stacked.loc[indexer] = mu[i] + dot(Sigma_expr, condvalues - mu[j])
            else:
                # special case: no categorical fields left. hence we cannot stack over then, it is only a single mu left
                # and we only need to update that
                self._mu = self._mu[i] + dot(Sigma_expr, condvalues - self._mu[j])

        # remove fields as needed
        self.fields = [field for field in self.fields if field['name'] not in remove]
        self._categoricals = [name for name in self._categoricals if name not in remove]
        self._numericals = [name for name in self._numericals if name not in remove]

        return self.update()

    def _marginalizeout(self, keep):
        # todo: factor out to models.py
        if len(keep) == self._n or self._isempty():
            return self

        keep = set(keep)
        num_keep = [name for name in self._numericals if name in keep]  # note: this is guaranteed to be sorted
        cat_remove = [name for name in self._categoricals if name not in keep]

        # use weak marginals to get the best approximation of the marginal distribution that is still a cg-distribution
        # clone old values
        p = self._p.copy()
        mu = self._mu.copy()

        # marginalized p just like in the categorical case (categoricals.py), i.e. sum up over removed dimensions
        self._p = self._p.sum(cat_remove)

        # marginalized mu (taken from the script)
        # slice out the gaussian part to keep; sum over the categorical part to remove
        if len(num_keep) != 0:
            #todo: does the above work in all cases? if len(self._numericals) != 0:
            mu = mu.loc[dict(mean=num_keep)]
            self._mu = (p * mu).sum(cat_remove) / self._p

            # marginalized sigma
            # TODO: this is kind of wrong... the best CG-approximation does not have a single S but a different one for each x in omega_X...
            # TODO: use numerical indices instead, then we don't need xarray for _S anymore and dealing with math becomes easier everywhere else
            self._S = self._S.loc[num_keep, num_keep]

        # update fields and dependent variabless
        self.fields = [field for field in self.fields if field['name'] in keep]
        self._categoricals = [name for name in self._categoricals if name in keep]
        self._numericals = num_keep

        return self.update()

    def _density(self, x):
        """Returns the density of the model at point x.

        Args:
            x: a list of values as input for the density.
        """
        cat_len = len(self._categoricals)
        num_len = len(self._numericals)
        cat = tuple(x[:cat_len])  # need it as a tuple for indexing below
        num = np.array(x[cat_len:])  # need as np array for dot product

        p = self._p.loc[cat].data

        if num_len == 0:
            return p

        # works because gaussian variables are after categoricals.
        # Therefore the only not specified dimension is the last one, i.e. the one that holds the mean!
        mu = self._mu.loc[cat].data

        # TODO: also remove matrix nonsense from everywhere else, including gaussians.py
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
        # TODO: ask Frank about it
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

    def _sample(self):
        raise NotImplementedError()

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._mu = self._mu.copy()
        mycopy._S = self._S.copy()
        mycopy._p = self._p.copy()
        mycopy._categoricals = self._categoricals.copy()
        mycopy._numericals = self._numericals.copy()
        mycopy.update()
        return mycopy

if __name__ == '__main__':
    # generate input data
    Sigma = np.diag([1, 1, 1])
    Sigma = np.matrix([[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]])
    Sigma = np.diag([1, 1, 1, 1])

    # select data set using this indicator variable
    #dataset = "dummy_cg"
    #dataset = "a"
    dataset = "b"

    if dataset == "a":
        n = 1000
        testopts = {'levels': {0: [1, 2, 3, 4], 1: [2, 5, 10], 2: [1, 2]},
                    'Sigma': Sigma,
                    'fun': genCatData,
                    'catvalasmean': 1,  # works if dc = dg
                    'seed': 10}
        data = genCGSample(n, testopts)  # categoricals first, then gaussians, np array
        dc = len(testopts.keys())
    elif dataset == "b":
        n = 1000
        testopts = {'levels': {0: [0, 1], 1: [0, 1], 2: [0, 1]},
                    'Sigma': Sigma,
                    'fun': genCatDataJEx,
                    'catvalasmean': 1,
                    'seed': 10}
        data = genCGSample(n, testopts)  # categoricals first, then gaussians, np array
        dc = len(testopts.keys())
        print("dc ", dc)
    elif dataset == "dummy_cg":
        data = ConditionallyGaussianModel.cg_dummy()

    # fit model
    model = ConditionallyGaussianModel('testmodel')
    model.fit(data)

    # print some information about the model
    print(model)
    print('p_ML: \n', model._p)
    print('mu_ML: \n', model._mu)
    print('Sigma_ML: \n', model._S)

    if dataset == "dummy_cg":
        md.Model.save(model, 'mymb.cg_dummy.mdl')
        copy = model.copy()

        print('p(M) = ', model._density(['M', 'Jena', 0, -6]))
        print('argmax of p(sex, city, age, income) = ', model._maximum())
        model.model(model=['sex', 'city', 'age'])  # marginalize income out
        print('p(M) = ', model._density(['M', 'Jena', 0]))
        print('argmax of p(sex, city, age) = ', model._maximum())
        model.model(model=['sex', 'age'], where=[('city', "==", 'Jena')])  # condition city out
        print('p(M) = ', model._density(['M', 0]))
        print('argmax of p(sex, agge) = ', model._maximum())
        model.model(model=['sex'], where=[('age', "==", 0)])  # condition age out
        print('p(M) = ', model._density(['M']))
        print('p(F) = ', model._density(['F']))
        print('argmax of p(sex) = ', model._maximum())

    # ind = (0, 1, 1)
    # print('mu(', [model._extents[i][ind[i]] for i in ind], '):', model._mu[ind])
    #print(np.histogram(data[:, dc]))
    plothist(data.iloc[:, dc + 1].ravel())
