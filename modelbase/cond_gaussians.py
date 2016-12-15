
import logging
import pandas as pd
import numpy as np
from numpy import pi, exp, matrix, ix_, nan

import utils
import models as md
from models import AggregationTuple, SplitTuple, ConditionTuple
import domains as dm

#imports frank
#from CGSNR_NLPs import GMNLP_GAUSS_SNR, GMNLP_CAT_SNR
from datasampling import genCGSample, genCatData, genCatDataJEx, genMixGSample
import xarray as xr

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def getCumlevelsAndDict(levels, excludeindex = None):
    """
    cumlevels[i] ... total # of levels of categorical variables with index <=i
    dval2ind[i][v] ... index of value v in the list of levels of variable i
    """
    dc = len(levels.keys())    
    
    dval2ind={}
    for i in range(dc):
        dval2ind[i] = {}
        for j, v in enumerate(levels[i]):
            dval2ind[i][v] = j # assign numerical value to each level of variable i
            
    cumlevels = [0]
    for v in levels.keys():
        cumlevels.append(cumlevels[-1]+len(levels[v]))

    
    return (cumlevels, dval2ind)
    
    
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
                data structure used: numpy array
            _mu:
                meaning: the mean of each conditional gaussian.
                data structure used: xarray DataArray with m+1 dimensions. The first m dimensions represent the
                categorical random variables of the model and are labeled accordingly. The last dimension represents
                the mean vector of the continuous part of the model and is therefore of length n.
    """

    def __init__(self, name):
        super().__init__(name)

        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }

    # @staticmethod
    # def _get_header(df):
    #     """ Returns suitable fields for this model from a given pandas dataframe.
    #     """
    #     fields = []
    #     for colname in df:
    #         column = df[colname]
    #         # if categorical of some sort, create discrete field from it
    #         if column.dtype == "category" or column.dtype == "object":
    #             domain = dm.DiscreteDomain()
    #             extent = dm.DiscreteDomain(sorted(column.unique()))
    #             field = md.Field(colname, domain, extent, 'string')
    #         # else it's numeric
    #         else:
    #             field = md.Field(colname, dm.NumericDomain(), dm.NumericDomain(column.min(), column.max()), 'numerical')
    #         fields.append(field)
    #     return fields

    def _fitFullLikelihood(self, data, fields, dc):
        """fit full likelihood for CG model"""
        n, d = data.shape
        dg = d - dc
        
        cols = data.columns
        catcols = cols[:dc]
        gausscols = cols[dc:]        
        
        extents = [f['extent'].value() for f in fields[:dc]] # levels
        sizes = [len(v) for v in extents]
#        print('extents:', extents)
    
        z = np.zeros(tuple(sizes))
        pML = xr.DataArray(data=z, coords=extents, dims=catcols)

        #### mus
        mus = np.zeros(tuple(sizes + [dg]))
        coords = extents + [[contname for contname in gausscols]]
        dims = catcols | [['mean']]
        musML = xr.DataArray(data=mus, coords=coords, dims=dims)
        
        # calculate p(x)
        for row in data.itertuples():
            cats = row[1:1+dc]
            gauss = row[1+dc:]

            pML.loc[cats] += 1
            musML.loc[cats] += gauss

#        print(pMLx)

#            
        it = np.nditer(pML, flags=['multi_index']) # iterator over complete array
        while not it.finished:
            ind = it.multi_index
#            print "%d <%s>" % (it[0], it.multi_index)
#            print(ind, pMLx[ind])
            musML[ind] /= pML[ind]
            it.iternext()
        pML /= 1.0 *n

        Sigma = np.zeros((dg, dg))
        for row in data.itertuples():
            cats = row[1:1+dc]
            gauss = row[1+dc:]
            ymu = np.matrix( gauss - musML.loc[cats])
            Sigma += np.dot(ymu.T, ymu)

        Sigma /= n
        
        return (pML, musML, Sigma)

    def fit(self, df):
        """Fits the model to passed DataFrame

        Parameters:
            df: A pandas data frame that holds the data to fit the model to. All columns in the data frame will be used.
                The data frame must have its columns ordered such that categorical columns occur before continuous
                columns.

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
            #if column.dtype == "category" or column.dtype == "object":
            if column.dtype == "object":
                categoricals.append(colname)
            else:
                numericals.append(colname)

        # reorder data frame such that categorical columns are first
        df = pd.DataFrame(df, columns=categoricals+numericals)

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

        # update field access by name
        self._update()
        
#        data = genCGSample(n, testopts) # categoricals first, then gaussians
        dg = len(numericals); dc = len(categoricals); d = dc + dg
        
        # get levels
        extents = [f['extent'].value() for f in fields[:dc]]

        
#        print(df[0:10])
        
        (p, mus, Sigma) = self._fitFullLikelihood(df, fields,  dc)

        print ('pML:', p)
        
        ind = (0,1,1)
        print('mu(',[extents[i][ind[i]] for i in ind], '):', mus[ind])
        
        print('Sigma:', Sigma)
        
    #    print(np.histogram(data[:, dc]))
#        plothist(data.iloc[:, dc+1].ravel())
        
        # @Frank:
        # - der data frame hat die kategorischen variables vorn, danach die kontinuierlichen
        # - categoricals und numericals enthält die namen der kategorischen/kontinuierlichen ZV

        # @Frank: generell um herauszufinden welche columns/fields kategorisch sind:
        #   - self.byname(name) liefert dir das 'Field' zu einer Zufallsvariable. Zu Field schau mal Zeile 72ff in models.py
        #   - df.dtypes gibt dir eine Liste mit dem Datentyp der Roh-Daten
        #   - df[colname].dtype gibt dir den Datentyp einer spez. column

        # TODO
#        raise NotImplementedError()
#        return self.update()

    @staticmethod
    def cg_dummy():
        """Returns a dataframe that contains sample of a 4d cg distribution. See the code for the used parameters."""
        # chose fixed parameters
        mu_M_Jena = [0, 0]
        mu_F_Jena = [1, 3]
        mu_M_Erfurt = [-10, 1]
        mu_F_Erfurt = [-5, -6]
        S = [[3, 0.5], [0.5, 1]]
        dims = ['sex', 'city', 'age', 'income']
        # and a sample size
        samplecnt = 200

        # generate samples for each and arrange in dataframe
        df_cat = pd.concat([
            pd.DataFrame([["M", "Jena"]] * samplecnt, columns=['sex', 'city']),
            pd.DataFrame([["F", "Jena"]] * samplecnt, columns=['sex', 'city']),
            pd.DataFrame([["M", "Erfurt"]] * samplecnt, columns=['sex', 'city']),
            pd.DataFrame([["F", "Erfurt"]] * samplecnt, columns=['sex', 'city'])
        ])

        df_num = pd.concat([
            pd.DataFrame(np.random.multivariate_normal(mu_M_Jena, S, samplecnt), columns=['age', 'income']),
            pd.DataFrame(np.random.multivariate_normal(mu_F_Jena, S, samplecnt), columns=['age', 'income']),
            pd.DataFrame(np.random.multivariate_normal(mu_M_Erfurt, S, samplecnt), columns=['age', 'income']),
            pd.DataFrame(np.random.multivariate_normal(mu_F_Erfurt, S, samplecnt), columns=['age', 'income'])
        ])
        df = pd.concat([df_num, df_cat], axis=1)
#        df.plot.scatter(x="age", y="income")
        return df


    def update(self):
        """Updates dependent parameters / precalculated values of the model after some internal changes."""
        self._update()
        raise NotImplementedError()
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

        raise NotImplementedError()

        self.fields = [self.fields[idx] for idx in i]
        return self.update()

    def _marginalizeout(self, keep):
        if len(keep) == self._n or self._isempty():
            return self
        if len(keep) == 0:
            return self._setempty()

        # i.e.: just select the part of mu and sigma that remains
        #keepidx = sorted(self.asindex(keep))
        # TODO
        raise NotImplementedError()
        #self.fields = [self.fields[idx] for idx in keepidx]
        return self.update()

    def _density(self, x):
        """Returns the density of the model at point x.

        Args:
            x: a list of values as input for the density.
        """
        raise NotImplementedError()

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        raise NotImplementedError()

    def _sample(self):
        raise NotImplementedError()

    def copy(self, name=None):
        name = self.name if name is None else name
        mycopy = ConditionallyGaussianModel(name)
        # todo: implement rest
        #raise NotImplementedError()
        #mycopy.update()
        #return mycopy


__philipp__ = False
if __name__ == '__main__':
#    import pdb

    Sigma = np.diag([1,1,1])
    Sigma = np.matrix([[1,0,0.5],[0,1,0],[0.5,0,1]])
#    Sigma = np.diag([1,1,1,1])
    
    independent = 1
    if independent:
        n = 1000
        testopts={'levels' :  {0: [1,2,3,4], 1: [2, 5, 10], 2: [1,2]}, 
        'Sigma' : Sigma, 
        'fun' : genCatData, 
        'catvalasmean' :  1, # works if dc = dg 
        'seed' : 10}
    else:
        n = 1000
        testopts={'levels' :  {0:[0,1], 1:[0,1], 2:[0,1]}, 
        'Sigma' :Sigma,
        'fun' : genCatDataJEx, 
        'catvalasmean' :  1, 
        'seed': 10}

    data = genCGSample(n, testopts) # categoricals first, then gaussians, np array
    
    
    
    model = ConditionallyGaussianModel('model1')
    
    model.fit(data)