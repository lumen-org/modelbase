# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:43:29 2016

@author: Philipp Lucas
"""

import pdb
import numpy as np
from numpy import pi, exp, matrix, ix_

import seaborn.apionly as sns

import sklearn as skl
from sklearn import mixture
   

    
''' 
 ## how to get from data to model ##
   
1. provide some data file
2. open that data file
3. read that file into a tabular structure and guess its header, i.e. its columns and its data types
4. use it to train a model
5. return the model

https://github.com/rasbt/pattern_classification/blob/master/resources/python_data_libraries.md !!!

## query a model

1. receive JSON object that describes the query
2. execute query: either do moddeling?



what about using a different syntax for the model, like the following:

    model['A'] : to select a submodel only on random variable with name 'A' // marginalize
    model['B'] = ...some domain...   // condition
    model.
'''


#LoadDataFromCsv    


def invertedIdxList (idx, len) :
    '''utility function that returns an inverted index list, e.g. given [0,1,4] and len=6 it returns [2,3,5].'''
    return list( set(range(0, len)) - set(idx) )   
    
def UpperSchurCompl (M, idx):
    '''Returns the upper Schur complement of matrix M with the 'upper block' indexed by i'''
    # derive correct index lists (using set opreations)
    i = idx
    j = invertedIdxList(i, M.shape[0])

    # that the definition of the upper Schur complement
    return M[ix_(i,i)] - M[ix_(i,j)] * M[ix_(j,j)].I * M[ix_(j,i)]        

    
class Field:    
    '''a random variable of a probability model'''
    def __init__ (self, label=None, domain=None, dtype=None, base=None):        
        if label is not None and domain is not None:
            # label of the field, i.e a string descriptor
            self.label = label
            # data type: either 'float' or 'categorical'
            self.dtype = dtype
            # range of possible values, either a tuple (dtype == 'categorial') or a numerical range as a tuple (min, max)
            self.domain = domain
        elif base is not None:
            raise NotImplementedError()
        else:
            raise ValueError()
    
        
class Model:
    '''an abstract base model that provides an interface to derive submodels from it or query density and other aggregations of it'''
    def _getHeader (df):
        # derive fields from data
        fields = []
        for column in df:
            ''' todo: this only works for continuous data '''
            field = Field( label = column, domain = (df[column].min(), df[column].max()), dtype = 'continuous' )
            fields.append(field)
        return fields
       
    def __init__ (self, name, dataframe):
            self.name = name
            self.data = dataframe
            self.fields = Model._getHeader(self.data)
            self._aggrMethods = None
            
    def fit (self):
        pass        
            
    def marginalize (self, keep = [], remove = []):
        if keep:
            self._marginalize(keep)
        else:
            raise NotImplementedError()    
    
    def _marginalize (self, keep):
        pass
    
    def condition (self, pairs):
        pass
    
    def aggregate (self, method):
        if (method in self._aggrMethods):
            return self._aggrMethods[method]()
        else:
            raise NotImplementedError()
    
    def copy(self):
        pass


class MultiVariateGaussianModel (Model):
    '''a multivariate gaussian model and methods to derive submodels from it or query density and other aggregations of it'''
    def __init__ (self, name = "iris", data = sns.load_dataset('iris').iloc[:, 0:4]):
        # make sure these are matrix types (numpy.matrix)
        super().__init__(name, data)              
        self._mu = np.nan
        self._S = np.nan        
        self._aggrMethods = {
            'argmax': self._argmax,
            'argavg': self._argmax
        }
    
    def fit (self):
        model = mixture.GMM(n_components=1, covariance_type='full')
        model.fit(self.data)
        self._mu = matrix(model.means_).T
        self._S = matrix(model.covars_)
        self._update()
    
    def summary (self):
        return( "Multivariate Gaussian Model '" + self.name + "':\n" + \
                "dimension:\n" + str(self._n) + "\n" + \
                "mu:\n" + str(self._mu) + "\n" + \
                "sigma:\n" + str(self._S) + "\n" )
        
    def _update (self):
        self._n = self._mu.shape[0]        
        self._detS = np.abs(np.linalg.det(self._S))
        self._SInv = self._S.I
        
    def condition (self, pairs):        
        i, xj = zip(*pairs)
        j = invertedIdxList(i, self._n)
        
        # store old sigma and mu
        S = self._S
        mu = self._mu
        
        # update sigma and mu according to GM script
        self._S = MultiVariateGaussianModel.UpperSchurCompl(S, i)        
        self._mu = mu[i] + S[ix_(i,j)] * S[ix_(j,j)].I * (xj - mu[j])
        self._update()
    
    def _marginalize (self, keep):
        # just select the part of mu and sigma that remains
        self._mu = self._mu[keep]  
        self._S = self._S[np.ix_(keep, keep)]
        self._update()
    
    def _density (self, x):   
        xmu = x - self._mu
        return (2*pi)**(-self._n/2) * (self._detS**-.5) * exp( -.5 * xmu.T * self._SInv * xmu )
        
    def _argmax (self):
        return self._mu
        
    def copy (self):
        mycopy = MultiVariateGaussianModel(name = self.name, data = self.data)
        mycopy._mu = self._mu
        mycopy._S = self._S
        mycopy._update()
        return mycopy
        
        
class ModelBase:
    '''a ModelBase is like a DataBase(-Management System): it holds models and allows queries against them'''
    def _loadIrisModel ():
        # load data set as pandas DataFrame
        data = sns.load_dataset('iris')                              

        # train model on continuous part of the data
        model = MultiVariateGaussianModel('iris', data.iloc[:, 0:-1])
        model.fit()        
        return model
        
    def _loadCarCrashModel ():        
        data = sns.load_dataset('car_crashes')
        model = MultiVariateGaussianModel('car_crashes', data.iloc[:, 0:-1])
        model.fit()
        return model

    def __init__ (self):
        
        # load some default models
        self.models = {}
        self.models['iris'] =  ModelBase._loadIrisModel()
        self.models['car_crashes'] = ModelBase._loadCarCrashModel()
    
    def execute (self, query):
        pass           
        
        
if __name__ == '__main__':
     mvg = MultiVariateGaussianModel()
     mvg.fit()
     print(mvg._density(np.matrix('1 1 1 1').T))