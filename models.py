"""
@author: Philipp Lucas

This module defines:

   * *var* Model: an abstract base class for models.
   * *var* Field: a class that represent random variables in a model.
   
It also defines models that implement that base model:
   
   * *var* MultiVariateGaussianModel
"""
import pandas as pd
import numpy as np
from numpy import pi, exp, matrix, ix_, nan
import copy as cp
from collections import namedtuple
from functools import reduce
from sklearn import mixture
import logging 
import seaborn.apionly as sns # probably remove this import later. Just for convenience to have default data for models available
import splitter as sp

# for fuzzy comparision. 
# TODO: make it nicer?
eps = 0.000001

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
  
""" Development Notes (Philipp)

## how to get from data to model ##
   
1. provide some data file
2. open that data file
3. read that file into a tabular structure and guess its header, i.e. its 
    columns and its data types
    * pandas dataframes can aparently guess headers/data types 
4. use it to train a model
5. return the model

https://github.com/rasbt/pattern_classification/blob/master/resources/python_data_libraries.md !!!

### what about using a different syntax for the model, like the following:

    model['A'] : to select a submodel only on random variable with name 'A' // marginalize
    model['B'] = ...some domain...   // condition
    model.
    
### other    
Somehow, I get the feeling I'm using numpy not correctly. it's too complicated
to always have to write matrix() explicitely 
"""

### UTILITY FUNCTIONS ###

def invertedIdxList (idx, len) :
    """utility function that returns an inverted index list, e.g. given [0,1,4]
    and len=6 it returns [2,3,5].
    """
    return list( set(range(0, len)) - set(idx) )   
    
def UpperSchurCompl (M, idx):
    """Returns the upper Schur complement of matrix M with the 'upper block' 
    indexed by i.
    """
    # derive index lists
    i = idx
    j = invertedIdxList(i, M.shape[0])
    # that's the definition of the upper Schur complement
    return M[ix_(i,i)] - M[ix_(i,j)] * M[ix_(j,j)].I * M[ix_(j,i)]        

### GENERIC / ABSTRACT MODELS and other base classes ###
AggregationTuple = namedtuple('AggregationTuple', ['name', 'method', 'args'])
SplitTuple = namedtuple('SplitTuple', ['name', 'method', 'args'])
ConditionTuple = namedtuple('ConditionTuple', ['name', 'operator', 'value'])
Field = namedtuple('Field', ['name', 'domain', 'dtype'])
"""a random variable of a probability model.

   name ... name of the field, i.e a string descriptor
   domain ... range of possible values, either a list (dtype == 'string') 
       or a numerical range as a tuple (min, max) (dtype == 'numerical')
   dtype ... data type: either 'numerical' or 'string'
"""

'''      
class Field(dict):    
    """a random variable of a probability model.
    
       name ... name of the field, i.e a string descriptor
       domain ... range of possible values, either a list (dtype == 'string') 
           or a numerical range as a tuple (min, max) (dtype == 'numerical')
       dtype ... data type: either 'numerical' or 'string'
    """
    def __init__ (self, name=None, domain=None, dtype=None):        
        # just a fancy way of providing a clean interface to actually nothing more than a python dict
        if (name is not None) and (domain is not None):
            super().__init__(name=name, domain=domain, dtype=dtype)
        else:
            raise ValueError("invalid argument values")
    
    def __str__ (self):
        return self['name'] + "(" + self['dtype'] + ")" '''
       
class Model:
    """An abstract base model that provides an interface to derive submodels
    from it or query density and other aggregations of it.
    """
    @staticmethod
    def _getHeader (df):
        """Returns suitable fields for a model from a given pandas dataframe. 
            
            TODO: at the moment this only works for continuous data.
        """
        fields = []
        for column in df:
            field = Field(name = column, domain = (df[column].min(), df[column].max()), dtype = 'numerical' )
            fields.append(field)
        return fields

    def _asIndex (self, names):
        """Given a single name or a list of names of random variables, returns
        the indexes of these in the .field attribute of the model.
        """
        if isinstance(names, str):
            return self._name2idx[names]
        else:
            return [self._name2idx[name] for name in names]
            
    def _byName (self, names):
        """Given a list of names of random variables, returns the corresponding
        fields of this model.
        """        
        if isinstance(names, str):
            return self.fields[self._name2idx[names]]
        else:
            return [self.fields[self._name2idx[name]] for name in names]
                    
    def isFieldName (self, names):
        """Returns true iff the name or names of variables given are names of 
        random variables of this model.
        """
        if isinstance(names, str):
            names = [names]
        return all([name in self._name2idx for name in names])
#        return all(map(lambda name: name in self._name2idx, names))       
#        return all(map(lambda name: any(map(lambda field: field.name == name, self.fields)), names))
       
    def __init__ (self, name):
        self.name = name          
        self._aggrMethods = None
        self.fields = []        
            
    def fit (self, data):
        """Fits the model to the dataframe assigned to this model in at 
        construction time.
        
        Returns:
            The modified model.
        """
        self.data = data
        self.fields = Model._getHeader(self.data)        
        self._fit()        
        return self
        
    def _fit (self):
        raise NotImplementedError()
            
    def marginalize (self, keep = None, remove = None):
        """Marginalizes random variables out of the model. Either specify which 
        random variables to keep or specify which to remove. 
        
        Note that marginalization is depending on the domain of a random 
        variable. That is: if nothing but a single value is left in the 
        domain it is conditioned on this value (and marginalized out). 
        Otherwise it is 'normally' marginalized out (assuming that the full 
        domain is available)
        
        Returns:
            The modified model.
        """        
        logger.debug('marginalizing: '
            + ('keep = ' + str(keep) if remove is None else ', remove = ' + str(remove)))
        
        if keep is not None:
            if not self.isFieldName(keep):
                raise ValueError("invalid random variable names: " + str(keep))
        elif remove is not None:
            if not self.isFieldName(remove):
                raise ValueError("invalid random variable names")
            keep = set([f.name for f in self.fields]) - set(remove)
        
        self._marginalize(keep)        
        return self
    
    def _marginalize (self, keep):
        raise NotImplementedError()
    
    def condition (self, pairs):
        """Conditions this model according to the list of 2-tuples 
        (<name-of-random-variable>, <condition-value>).
        
        Note: This simply restricts the domains of the random variables. To 
        remove the conditioned random variable you need to call marginalize
        with the appropiate paramters.
        
        Returns:
            The modified model.
        """
        for (name, value) in pairs:            
            if not self.isFieldName(name):
                raise ValueError(name + " is not a name of a field in the model")
            randVar = self._byName(name)
            if ((randVar.dtype == "string" and value not in randVar.domain) or 
                (randVar.dtype == "numerical" and (value + eps < randVar.domain[0] or value - eps > randVar.domain[1]))):
                raise ValueError("the value to condition on is not in the domain of random variable " + name)
        self._condition(pairs)
        return self
    
    def _condition (self, pairs):
        raise NotImplementedError()
    
    def aggregate (self, method):
        """Aggregates this model using the given method and returns the 
        aggregation as a list. The order of elements in the list, matches the 
        order of random variables in the models field.
        
        Returns:
            The aggregation of the model.
        """
        if (method in self._aggrMethods):
            return self._aggrMethods[method]()
        else:
            raise NotImplementedError("Your Model does not provide the requested aggregation '" + method + "'")     
       
    def density(self, names, values=None):
        """Returns the density at given point. You may either pass both, names
        and values, or only one list with values. In the latter case values is
        assumed to be in the same order as the fields of the model.
        """
        if values is None:
            # in that case the only argument holds the (correctly sorted) values
            values = names 
        else:
            sorted_ = sorted(zip(self._asIndex(names), values), lambda pair: pair[0])
            values = [pair[0] for pair in sorted_]
        return self._density(values)
            
    def sample (self, n=1):
        """Returns n samples drawn from the model."""
        samples = (self._sample() for i in range(n))
        return  pd.DataFrame.from_records(samples, self.names)        

    def _sample(self):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
        
    def _update(self):
        """Updates the name2idx dictionary based on the fields in .fields"""    
# TODO": call it from aggregate, ... make it transparent to subclasses!? is that possible?    
        self._name2idx = dict(zip([f.name for f in self.fields], range(len(self.fields))))
        self.names = [f.name for f in self.fields]

    def model (self, model, where, as_ = None):
        """Returns a model with name 'as_' that models the fields in 'model'
        respecting conditions in 'where'. 
        
        Note that it does NOT create a copy, but modifies this model.
        
        Args:
            model:  A list of strings, representing the names of fields to model. 
            where: A list of 'conditiontuple's, representing the conditions to
                model. 
            as_: A string. The name for the model to derive. If set to None the
                name of the base model is used.
        
        Returns:
            The modified model.
        """
        self.name = self._name if as_ is None else as_        
        # 1. copy model 
        #derivedModel = self.copy(name = as_)
        # 2. apply filter, i.e. condition
        equalpairs = [(cond.name, cond.value) for cond in where if cond.operator == 'EQUALS']        
        # 3. + 4. + 5: condition, marginalize and return
        return derivedModel.condition(equalpairs).marginalize(keep = model)
    
    def predict (self, predict, where=[], splitby=[], returnbasemodel = False):
        """ Calculates the prediction against the model and returns its result
        by means of a data frame.
        
        The data frame contains exactly those columns/random variables which
        are specified in 'predict'. Its order is preserved.
        
        Args:
            predict: A list of names of fields (strings) and 'AggregationTuple's. 
                This is hence the list of fields to be included in the returned 
                dataframe.
            where: A list of filters to use. The list consists of 'ConditionTuple's.
            splitby: A list of 'SplitTuple's, i.e. a list of fields on which to 
                split the model and the method how to do the split.
            returnbasemodel: A boolean flag. If set this method will return a pair 
                constisting of the dataframe and the basemodel for the prediction.
                Defaults to False.
        Returns:
            A dataframe with the fields as given in 'predict', or a tuple (see 
            parameter returnbasemodel).
        """       
        # (1) derive the base model,
        # i.e. a model on all requested dimensions and measures, respecting filters 
        # TODO: is there any filter that cannot be applied yet?        
        
        # derive the list of dimensions to split by
        split_names = [f.name for f in splitby]        
        # derive the list of aggregations and dimensions to include in result table        
        aggrs, aggr_names, dim_names, predict_names = [], [], [], []
        for f in predict:
            if isinstance(f, str):
                # f is just a string, i.e name of a field            
                dim_names.append(f)
                predict_names.append(f)
            else:
                name = f.name
                predict_names.append(name)
                aggr_names.append(name)
                aggrs.append(f)
            
        # from that derive the set of (names of) random variables that are to be kept for the base model
        basenames = list(set(split_names) | set(aggr_names) | set(dim_names))
        # now get the base model    
        basemodel = self.copy().model(basenames, where, '__' + self.name + '_base')
        
        # (2) derive a sub-model for each requested aggregation
        # i.e. remove all random variables of other measures which are not also a used for splitting
        # or equivalently: keep all random variables of dimensions, plus the once for the current aggregation
        splitnames_unique = set(split_names) # WHICH ARE NOT ALSO USED AS A DIMENSION
        i = 0
        def _derive_aggregation_model (aggr_name):
            nonlocal i
            model = self.copy().model(
                model = list(splitnames_unique | set([aggr_name])),
                as_ = basemodel.name + "_" + aggr_name + str(i))
            i+=1
            return model
        # TODO: use a different naming scheme later, e.g.: name = _id_generator(),                
        aggr_models = [_derive_aggregation_model(name) for name in aggr_names]
        
        # TODO: derive model for density
        # TODO: is density really just another aggregation?        
        
        # (3) generate input for model aggregations,
        # i.e. a cross join of splits of all dimensions
        # note: filters on dimensions should already have been applied
        def _get_group_frame (split):
            MYMAGICNUMBER = 3
            name = split.name
            domain = basemodel._byName(name).domain
            domain = sp.NumericDomain(domain[0], domain[1])
            splitFct = sp.splitter[split.method]
            frame = pd.DataFrame( splitFct(domain, MYMAGICNUMBER), columns = [name])
            frame['__crossIdx__'] = 0 # need that index to crossjoin later
            return frame
        def _crossjoin (df1, df2):
            return pd.merge(df1, df2, on='__crossIdx__', copy=False)
        group_frames = map(_get_group_frame, splitby)
        input_frame = reduce(_crossjoin, group_frames, next(group_frames)).drop('__crossIdx__', axis=1)
                            
        # (4) query models and fill result data frme
        """ question is: how to efficiently query the model? how can I vectorize it?
            I believe that depends on the query. A typical query is consists of
            dimensions for splits and then aggregations and densities. 
            For the case of aggregations a new conditioned model has to be 
            calculated for every split. I don't see how to vectorize / speed
            this up easily.
            For densities it might be very well possible, as the split are
            now simply input to some density function.
        """
                            
        # just start simple: don't do it vectorized! 
        # TODO: it might actually be faster to first condition the model on the
        # dimensions (values) and then derive the measure models... 
        result_list = [input_frame]
        for idx, aggr in enumerate(aggrs):
            aggr_results = []
            aggr_model = aggr_models[idx]
            for row in input_frame.iterrows():
                # derive model for these specific conditions...
                pairs = zip(split_names, row[1])
                mymodel = aggr_model.copy().condition(pairs).marginalize(keep = [aggr.name])
                # now do the aggregation
                # TODO: in the future, there may be multidimensional aggregations
                # for now, it's just 1d --> [0]
                res = mymodel.aggregate(aggr.method)[0]                
                aggr_results.append(res)
            series = pd.Series(aggr_results, name=aggr.name)
            result_list.append(series)
               
        # (5) filter on aggregations?
        # TODO?
        
        # (6) collect all into one data frame
        return_frame = pd.concat(result_list, axis=1)
                
        # (7) return correctly ordered frame that only contain requested variables
        # TOOD: this is buggy for the case of a field being returned multiple times...
        return return_frame[predict_names]

### ACTUAL MODEL IMPLEMENTATIONS ###

class MultiVariateGaussianModel (Model):
    """A multivariate gaussian model and methods to derive submodels from it
    or query density and other aggregations of it
    """
    def __init__ (self, name):
        # make sure these are matrix types (numpy.matrix)
        super().__init__(name)
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
        """updates dependent parameters / precalculated values of the model"""
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
            self._byName(pair[0]).domain = pair[1]
                
    def _conditionAndMarginalize (self, names):
        """Conditions the random variables with name in names on their 
        available domain and marginalizes them out
        """
        if len(names) == 0:
            return
        j = self._asIndex(names)
        i = invertedIdxList(j, self._n)
        condValues = [self.fields[idx].domain for idx in j]
        # store old sigma and mu
        S = self._S
        mu = self._mu
        # update sigma and mu according to GM script
        self._S = UpperSchurCompl(S, i)
        self._mu = mu[i] + S[ix_(i,j)] * S[ix_(j,j)].I * (condValues - mu[j])
        self.fields = [self.fields[idx] for idx in i]
        self.update()
    
    def _marginalize (self, keep):                
        # there is two types of random variable v that are removed: 
        # (i) v's domain is a single value, i.e. they are 'conditioned out'
        # (ii) v's domain is a range (continuous random variable) or a set
        #   (discrete random variable), i.e. they are 'normally' marginalized out
        
        # "is not tuple" means it must be a scalar value, hence a random variable 
        # to condition on for marginalizing it out # TODO: this is ugly, hard to 
        # read and maybe even slow
        condNames = [randVar.name for idx, randVar in enumerate(self.fields)
            if (randVar.name not in keep) and (type(randVar.domain) is not tuple)]
        self._conditionAndMarginalize(condNames)
        
        # marginalize all other not wanted random variables
        # i.e.: just select the part of mu and sigma that remains
        keepIdx = self._asIndex(keep)
        self._mu = self._mu[keepIdx]  
        self._S = self._S[np.ix_(keepIdx, keepIdx)]
        self.fields = [self.fields[idx] for idx in keepIdx]
        self.update()
    
    def _density (self, x):   
        """Returns the density of the model at point x."""
        xmu = x - self._mu
        return (2*pi)**(-self._n/2) * (self._detS**-.5) * exp( -.5 * xmu.T * self._SInv * xmu )
        
    def _maximum (self):
        # _mu is a np matrix, but I want to return a list
        return self._mu.tolist()[0] 
    
    def _sample  (self):
        # TODO: let it return a dataframe
        return self._S * np.matrix(np.random.randn(self._n)).T + self._mu
        
    def copy (self, name = None):
        name = self.name if name is None else name
        mycopy = MultiVariateGaussianModel(name)
        mycopy.data = self.data
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy._mu = self._mu
        mycopy._S = self._S
        mycopy.update()
        return mycopy
        