"""
@author: Philipp Lucas

This module defines:

   * *var* Model: an abstract base class for models.
   * *var* Field: a class that represent random variables in a model.
   
It also defines models that implement that base model:
   
   * *var* MultiVariateGaussianModel
"""
import copy as cp
import numpy as np
from numpy import pi, exp, matrix, ix_, nan
from sklearn import mixture
import logging 
import seaborn.apionly as sns # probably remove this import later. Just for convenience to have default data for models available

# for fuzzy comparision. 
# TODO: make it nicer?
eps = 0.000001

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
  
''' Development Notes (Philipp)

## how to get from data to model ##
   
1. provide some data file
2. open that data file
3. read that file into a tabular structure and guess its header, i.e. its columns and its data types
    * pandas dataframes can aparently guess headers/data types 
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

### UTILITY FUNCTIONS ###

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

### GENERIC / ABSTRACT MODELS and other base classes ###
      
class Field(dict):    
    '''a random variable of a probability model.
    
       name ... name of the field, i.e a string descriptor
       domain ... range of possible values, either a list (dtype == 'string') or a numerical range as a tuple (min, max) (dtype == 'numerical')
       dtype ... data type: either 'numerical' or 'string'
    '''
    def __init__ (self, name=None, domain=None, dtype=None):        
        # just a fancy way of providing a clean interface to actually nothing more than a python dict
        if (name is not None) and (domain is not None):
            super().__init__(name=name, domain=domain, dtype=dtype)
        else:
            raise ValueError("invalid argument values")
    
    def __str__ (self):
        return self['name'] + "(" + self['dtype'] + ")" 
       
class Model:
    '''an abstract base model that provides an interface to derive submodels from it or query density and other aggregations of it'''
    
    def _getHeader (df):
        ''' derive fields from a given pandas dataframe. 
            
            TODO: at the moment this only works for continuous data '''
        fields = []
        for column in df:
            field = Field(name = column, domain = (df[column].min(), df[column].max()), dtype = 'numerical' )
            fields.append(field)
        return fields

    def _asIndex (self, names):
        '''given a single name or a list of names of random variables, returns the indexes of these in the .field attribute of the model'''                
        if type(names) is not list:
            return self._name2idx[names]
        else:
            return [self._name2idx[name] for name in names]
            
    def _byName (self, names):
        '''given a list of names of random variables, returns the corresponding fields of this model'''        
        if type(names) is not list:
            return self.fields[self._name2idx[names]]
        else:
            return [self.fields[self._name2idx[name]] for name in names]
                    
    def _isRandomVariableName (self, names):
        '''Returns true iff the name or names of variables given are names of random variables of this model'''
        if type(names) is not list:
            names = [names]
#        return all(map(lambda name: name in self._name2idx, names))
        
        return all(map(lambda name: any(map(lambda field: field["name"] == name, self.fields)), names))
       
    def __init__ (self, name, dataframe):
        self.name = name
        self.data = dataframe                
        self._aggrMethods = None
        self.field = []        
            
    def fit (self):
        '''fits the model to the dataframe assigned to this model in at construction time'''
        self.fields = Model._getHeader(self.data)        
        self._fit()
        
    def _fit (self):
        raise NotImplementedError()
            
    def marginalize (self, keep = None, remove = None):
        '''Marginalizes random variables out of the model. Either specify which random variables to keep
        or specify which to remove. 
        
        Note that marginalization is depending on the domain of a random 
        variable. That is: if nothing but a single value is left in the 
        domain it is conditioned on this value (and marginalized out). 
        Otherwise it is 'normally' marginalized out (assuming that the full 
        domain is available)'''
        
        logger.debug('marginalizing: keep = ' + str(keep) + ', remove = ' + str(remove) )
        
        if keep is not None:
            if not self._isRandomVariableName(keep):
                raise ValueError("invalid random variable names: " + str(keep))
            self._marginalize(keep)
        elif remove is not None:
            if not self._isRandomVariableName(remove):
                raise ValueError("invalid random variable names")
# CHANGED            #keep = list( set( map(lambda f: f["name"], self.fields)) - set(remove) )
            keep = set( map(lambda f: f["name"], self.fields)) - set(remove)
            self._marginalize(keep)        
    
    def _marginalize (self, keep):
        raise NotImplementedError()
    
    def condition (self, pairs):
        '''conditions this model according to the list of 2-tuples 
        (<name-of-random-variable>, <condition-value>).
        
        Note: This simply restricts the domains of the random variables. To 
        remove the conditioned random variable you
        need to call marginalize with the appropiate paramters'''
        for (name, value) in pairs:            
            if not self._isRandomVariableName(name):
                raise ValueError("")
            randVar = self._byName(name)
            if ((randVar["dtype"] == "string" and value not in randVar["domain"]) or 
                (randVar["dtype"] == "numerical" and (value + eps < randVar["domain"][0] or value - eps > randVar["domain"][1]))):
                raise ValueError("the value to condition on is not in the domain of random variable " + name)
        self._condition(pairs)
    
    def _condition (self, keep):
        raise NotImplementedError()
    
    def aggregate (self, method):
        '''aggregates this model using the given method'''
        if (method in self._aggrMethods):
            return self._aggrMethods[method]()
        else:
            raise NotImplementedError("Your Model does not provide the requested aggregation '" + method + "'")
            
    def sample (self, n=1):
        '''returns n many samples drawn from the model'''
        return [self._sample() for i in range(n)]

    def _sample(self):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
        
    def _update(self):
        '''updates the name2idx dictionary based on the fields in .fields'''        
        self._name2idx = dict(zip([f['name'] for f in self.fields], range(len(self.fields))))

### ACTUAL MODEL IMPLEMENTATIONS ###

class MultiVariateGaussianModel (Model):
    '''a multivariate gaussian model and methods to derive submodels from it or query density and other aggregations of it'''
    def __init__ (self, name = "iris", data = sns.load_dataset('iris').iloc[:, 0:4]):
        # make sure these are matrix types (numpy.matrix)
        super().__init__(name, data)              
        self._mu = nan
        self._S = nan
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
    
    def _fit (self):
        model = mixture.GMM(n_components=1, covariance_type='full')
        model.fit(self.data)
        self._model = model        
        self._mu = matrix(model.means_).T
        self._S = matrix(model.covars_)
        self.update()
        
    def __str__ (self):
        return( "Multivariate Gaussian Model '" + self.name + "':\n" + \
                "dimension: " + str(self._n) + "\n" + \
                "random variables: " + str( [str(field) for field in self.fields] ))
#                "mu:\n" + str(self._mu) + "\n" + \
#               "sigma:\n" + str(self._S) + "\n")

    def update (self):
        '''updates dependent parameters / precalculated values of the model'''
        self._n = self._mu.shape[0]
        if self._n == 0:
            self._detS = None
            self._SInv = None
        else:
            self._detS = np.abs(np.linalg.det(self._S))
            self._SInv = self._S.I
        self._update()
       
    def _condition (self, pairs):
        for pair in pairs:
            self._byName(pair[0])["domain"] = pair[1]
                
    def _conditionAndMarginalize (self, names):
        '''conditions the random variables with name in names on their available domain and marginalizes them out'''
        if len(names) == 0:
            return        
        i = self._asIndex(names)
        j = invertedIdxList(i, self._n)  
        condValues = [self.fields[idx]["domain"] for idx in i]
        # store old sigma and mu
        S = self._S
        mu = self._mu                
        # update sigma and mu according to GM script
        self._S = UpperSchurCompl(S, i)        
        self._mu = mu[i] + S[ix_(i,j)] * S[ix_(j,j)].I * (condValues - mu[j])   
        self.fields = [self.fields[idx] for idx in j]
        self.update()        
    
    def _marginalize (self, keep):                
        # there is two types of random variable v that are removed: 
        # (i) v's domain is a single value, i.e. they are 'conditioned out'
        # (ii) v's domain is a range (continuous random variable) or a set (discrete random variable), i.e. they are 'normally' marginalized out
        
        # "is not tuple" means it must be a scalar value, hence a random variable to condition on for marginalizing it out
        condNames = [randVar["name"] for idx, randVar in enumerate(self.fields) if (randVar["name"] not in keep) and (type(randVar["domain"]) is not tuple)]
        self._conditionAndMarginalize(condNames)
        
        # marginalize all other not wanted random variables
        # i.e.: just select the part of mu and sigma that remains
        keepIdx = self._asIndex(keep)
        self._mu = self._mu[keepIdx]  
        self._S = self._S[np.ix_(keepIdx, keepIdx)]
        self.fields = [self.fields[idx] for idx in keepIdx]
        self.update()
    
    def _density (self, x):   
        xmu = x - self._mu
        return (2*pi)**(-self._n/2) * (self._detS**-.5) * exp( -.5 * xmu.T * self._SInv * xmu )
        
    def _maximum (self):
        return self._mu
    
    def _sample  (self):
        return self._S * np.matrix(np.random.randn(self._n)).T + self._mu
        
    def copy (self, name = None):        
        mycopy = MultiVariateGaussianModel(name = (self.name if name is None else name), data = self.data)
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy._mu = self._mu
        mycopy._S = self._S
        mycopy.update()
        return mycopy
