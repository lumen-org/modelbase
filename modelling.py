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
2. execute query:

### what about using a different syntax for the model, like the following:

    model['A'] : to select a submodel only on random variable with name 'A' // marginalize
    model['B'] = ...some domain...   // condition
    model.
    
### other    
Somehow, I get the feeling I'm using numpy not correctly. it's too complicated to always have to write matrix() explicitely 
'''

def invertedIdxList (idx, len) :
    '''utility function that returns an inverted index list, e.g. given [0,1,4] and len=6 it returns [2,3,5].'''
    return list( set(range(0, len)) - set(idx) )   
    
def UpperSchurCompl (M, idx):
    '''Returns the upper Schur complement of matrix M with the 'upper block' indexed by i'''
    # derive index lists
    i = idx
    j = invertedIdxList(i, M.shape[0])

    # that's the definition of the upper Schur complement
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
        ''' derive fields from data'''
        fields = []
        for column in df:
            ''' todo: this only works for continuous data '''
            field = Field( label = column, domain = (df[column].min(), df[column].max()), dtype = 'continuous' )
            fields.append(field)
        return fields

    def _asIndex (self, names):
        #return sorted( map( lambda name: self.fields))        
        #return sorted( map( lambda i: self.fields[i], names ) )        
        #return [ for name in names if self.fields.index()]        
        indices = []
        for idx, field in enumerate(self.fields):
            if field.label in names:
                indices.push(idx)
        return indices        
       
    def __init__ (self, name, dataframe):
        self.name = name
        self.data = dataframe
        self.fields = Model._getHeader(self.data)
        self._aggrMethods = None
            
    def fit (self):
        raise NotImplementedError()        
            
    def marginalize (self, keep = [], remove = []):
        if keep:
            self._marginalize(keep)
        else:
            raise NotImplementedError()    
    
    def _marginalize (self, keep):
        raise NotImplementedError()
    
    def condition (self, pairs):
        raise NotImplementedError()
    
    def aggregate (self, method):
        if (method in self._aggrMethods):
            return self._aggrMethods[method]()
        else:
            raise NotImplementedError()
            
    def sample (self, n=1):
        '''returns n many samples drawn from the model'''
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()


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
        self._model = model        
        self._mu = matrix(model.means_).T
        self._S = matrix(model.covars_)
        self._update()
    
    def describe (self):
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
    
    def sample  (self, n=1):
        return self._S * np.matrix(np.random.randn(self._n)).T + self._mu
        
    def copy (self):
        mycopy = MultiVariateGaussianModel(name = self.name, data = self.data)
        mycopy._mu = self._mu
        mycopy._S = self._S
        mycopy._update()
        return mycopy
        
        
class QuerySyntaxError(Exception):
    '''This error indicates that a PQL query was incomplete and hence could not be executed'''    
    meaning = 'This error indicates that a PQL query was incomplete and hence could not be executed'
    
    def __init__(self, message="", value=None):
        self.value = value
        self.message = message
        
    def __str__(self):
        return repr(self.value)
        

class QueryValueError(Exception):
    meaning = 'This error indicates that a PQL query contains a value that is semantically invalid, such as referring to a model that does not exist.'

    def __init__(self, message="", value=None):
        self.value = value
        self.message = message
    def __str__(self):
        return repr(self.value)
    
        
class ModelBase:
    '''a ModelBase is the analogon of a DataBase(-Management System) but for models: it holds models and allows queries against them'''
    def __init__ (self):        
        # load some default models
        # more data sets here: https://github.com/mwaskom/seaborn-data
        self.models = {}
        self.models['iris'] =  ModelBase._loadIrisModel()
        self.models['car_crashes'] = ModelBase._loadCarCrashModel()
    
    def execute (self, query):
        '''executes the given query on this model base'''
        # what's the command?
        if 'MODEL' in query:
            # do basic syntax and semantics checking of the given query
            if 'FROM' not in query:
                raise QuerySyntaxError("'FROM'-statement missing")
            if 'AS' not in query:
                raise QuerySyntaxError("'AS'-statement missing")            
            if 'WHERE' not in query:
                raise QuerySyntaxError("'FROM'-statement missing")                
            if query['FROM'] not in self.models:
                raise QueryValueError("The specified model does not exist.")
            
            self._model(randVars = list( map( lambda v: v['randVar'], query['MODEL'] ) ), 
                        baseModel = self.models[query['FROM']], 
                        name = query['AS'], 
                        filters = query.get('WHERE') )

        elif 'PREDICT' in query:
            raise NotImplementedError()
        
        elif 'DROP' in query:
            modelToDrop = query['DROP']
            self._drop(modelToDrop)

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
       
    def _add  (self, model, name):
        if name in self.models:
            pass
        self.models[name] = model
    
    def _drop (self, name):
        del self.models[name]
        
    def _model (self, randVars, baseModel, name, filters=None):
        # 1. copy model        
        derivedModel = baseModel.copy()        
        randVarIdxs = derivedModel._asIndex(randVars)
        # 2. apply filter, i.e. condition
        if filters is not None:
            raise NotImplementedError()
        # 3. remove unneeded random variables
        derivedModel._marginalize(keep = randVarIdxs)        
        # 4. store model in model base
        self._add(derivedModel, name)        
        
    def _predict (self, query):
        raise NotImplementedError()
        
if __name__ == '__main__':
     mvg = MultiVariateGaussianModel()
     mvg.fit()
     print(mvg._density(np.matrix('1 1 1 1').T))
     mb = ModelBase()
     cc = mb.models['car_crashes']