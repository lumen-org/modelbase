"""
@author: Philipp Lucas

The modelbase module primarly provides the ModelBase class.

Idiomatically the model base does the following:  
  * recieve a query
  * parse the query
  * execute the query  
"""

import logging
import string
import random
import pandas as pd

from functools import reduce
import seaborn.apionly as sns

import models as gm
# cross join for pandas data frams
from crossjoin import crossjoin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QuerySyntaxError(Exception):
    meaning = """This error indicates that a PQL query was incomplete and hence could not be executed"""
    
    def __init__(self, message="", value=None):
        self.value = value
        self.message = message
        
    def __repr__(self):
        return repr(self.value)
        

class QueryValueError(Exception):    
    meaning = """This error indicates that a PQL query contains a value that is semantically invalid, such as referring to a model that does not exist."""

    def __init__(self, message="", value=None):
        self.value = value
        self.message = message
        
    def __repr__(self):
        return repr(self.value)

ReturnCode = {
    "SUCCESS" : "success",
    "FAIL": "fail"
}

def _loadIrisModel ():
    """ Loads the iris data set and returns a MultiVariateGaussianModel of it"""    
    # load data set as pandas DataFrame
    data = sns.load_dataset('iris')
    # train model on continuous part of the data
    model = gm.MultiVariateGaussianModel('iris', data.iloc[:, 0:-1])
    model.fit()        
    return model
    
def _loadCarCrashModel ():     
    """ Loads the car crash data set and returns a MultiVariateGaussianModel of it"""
    data = sns.load_dataset('car_crashes')
    model = gm.MultiVariateGaussianModel('car_crashes', data.iloc[:, 0:-1])
    model.fit()
    return model

def _id_generator(size=15, chars=string.ascii_letters + string.digits, prefix='__'):
    """ Returns a prefixed, random string of letters and digits """
    return prefix + ''.join(random.choice(chars) for _ in range(size))    

class ModelBase:
    """A ModelBase is the analogon of a DataBase(-Management System) for
    models: it holds models and allows PQL queries against them.
    
    A Modelbase provides only one 'public' method:
        ModelBase.execute: It executes a given PQL query. All input and output 
            is provided as JSON objects, or simple strings / numbers if applicable.
       
    Furthermore it provides to 'public' attributes: 
        ModelBase.name: the id/name of this modelbase
        ModelBase.models: a dictionary of all models in the modelbase. Each 
            model must have a unique string as its name, and this name is used 
            as a key in the dictionary.
    """
    def __init__ (self, name):
        """ Creates a new instance and loads some default models. """
        # more data sets here: https://github.com/mwaskom/seaborn-data
        self.name = name        
        self.models = {} # models is a dictionary, using the name of a model as its key
        self.models['iris'] =  _loadIrisModel()
        self.models['car_crashes'] = _loadCarCrashModel()
        
    def __repr__ (self):
        return " -- Model Base > " + self.name+ " < -- \n" + \
            "contains " + str(len(self.models)) + " models, as follows:\n\n" + \
            reduce(lambda p, m: p + str(m) + "\n\n", self.models.values(), "")

###   _extract* functions are helpers to extract a certain part of a PQL query and do some basic syntax and semantic checks
    def _extractFrom (self, query):
        """ Returns the model that the value of the "FROM"-statement of query
        refers to. """
        if 'FROM' not in query:
            raise QuerySyntaxError("'FROM'-statement missing")
        modelName = query['FROM']
        if modelName not in self.models:
            raise QueryValueError("The specified model does not exist: " + modelName)            
        return self.models[modelName]
    
    def _extractShow (self, query):
        """ Extracts the value of the "SHOW"-statement from query."""        
        if 'SHOW' not in query:
            raise QuerySyntaxError("'SHOW'-statement missing")
        what = query['SHOW']
        if what not in ["HEADER", "MODELS"]:
            raise QueryValueError("Invalid value of SHOW-statement: " + what)
        return what
        
    def _extractGroupBy (self, query, required=False):    
        """ Extracts from query the list of conditions in the statement. The 
        order is preserved.
        
        Args:
            required: Optional flag. Set to True to allow for a missing 
                "WHERE" statement in the query.
            
        Returns:
            The list of names to group by or an empty list.
        """
        if required and 'GROUP BY' not in query:
            raise QuerySyntaxError("'GROUP BY'-statement missing")
        elif not required and 'GROUP BY' not in query:
            return []            
        return list( map( lambda v: v['randVar'], query['GROUP BY'] ) )
            
    def _extractModel (self, query):
        """ Extracts the names of the random variables to model and returns it
        as a list of strings. The order is preserved.
        
        Note that it returns only strings, not actual Field objects.
        
        TODO: internally this function refers to self.models[query['FROM']]. 
            Remove this dependency
        """
        if 'MODEL' not in query:
            raise QuerySyntaxError("'MODEL'-statement missing")            
        if query['MODEL'] == '*':
            return list( map( lambda field: field["name"], self.models[query['FROM']].fields ) )
        else:            
            return list( map( lambda v: v['randVar'], query['MODEL'] ) )
        
    def _extractPredict (self, query): 
        """ Extracts from query the fields to predict and returns them in a list. The order
        is preserved.
        
        The Fields are dicts that are identical to the JSON in the query.        
        """
        if 'PREDICT' not in query:
            raise QuerySyntaxError("'PREDICT'-statement missing")
        # return in one list                    
        return query['PREDICT']        
        
    def _extractAs (self, query):        
        """ Extracts from query the name under which the derived model is to be
        stored and returns it.
        """
        if 'AS' not in query:
            raise QuerySyntaxError("'AS'-statement missing")
        return query['AS']
        
    def _extractWhere (self, query, required=False):
        """ Extracts from query the list of conditions in the statement. The 
        order is preserved.
        
        Args:
            required: Optional flag. Set to True to allow for a missing 
                "WHERE" statement in the query.
            
        Returns:
            The list of conditions or an empty list.
        """
        if required and 'WHERE' not in query:
            raise QuerySyntaxError("'WHERE'-statement missing")
        elif not required and 'WHERE' not in query:
            return []            
        else:
            return query['WHERE']
    
    def execute (self, query):
        """ Executes the given PQL query and returns a status code and the 
        result (or None)
        
        Args:
            query: A word of the PQL language.
            
        Returns:
            The result of the query, which is some natively to JSON convertible
            object. I.e. '''json.dumps(result)''' works just fine.
        
        Raises:
            QuerySyntaxError: If there is some syntax error.
            QueryValueError: If there is some problem with a value of the query.
        """
        
        # basic syntax and semantics checking of the given query is done in the _extract* methods
        if 'MODEL' in query:            
            self._model( randVars = self._extractModel(query),
                        baseModel = self._extractFrom(query),
                        name = self._extractAs(query), 
                        filters = self._extractWhere(query) )
            return None

        elif 'PREDICT' in query:
            result = self._predict( aggrRandVars = self._extractPredict(query),
                    model = self._extractFrom(query),
                    filters = self._extractWhere(query),
                    groupBy = self._extractGroupBy (query) )
            return result
        
        elif 'DROP' in query:
            self._drop(name = query['DROP'])
            return None
            
        elif 'SHOW' in query:
            header = self._show( query = query, show = self._extractShow(query))
            return header            
       
    def _add  (self, model, name):
        """ Adds a model to the model base using the given name. """
        if name in self.models:
            logger.warn('Overwriting existing model in model base: ' + name)
        self.models[name] = model
        return model
    
    def _drop (self, name):
        """ Drops a model from the model base and returns the dropped model.
            
        Returns:
            None if there is no model with that name.
        """
        if name in self.models:
            model = self.models[name]
            del self.models[name]
            return model
        else:
            return None
        
    def _model (self, randVars, baseModel, name, filters=[], persistent = True):
        """ Runs a 'model' query against the model base and returns the resulting model.
        
        Args:
            randVars: A list of strings, representing the names of random 
                variables to model.
            baseModel: the model of which to derive the new model from.
            filters: Optional list of filters.
            persistent: Optional flag that controls whether or not the model 
                will actually cause any modification or addition to the model 
                base, i.e. if persistent is set to false, the requested model
                will still be generated and returned, but not added to the 
                model base itself.
                
        Returns:
            The derived model.
            
        Raises:
            An error if something went wrong.
        """        
        overwrite = baseModel.name == name 
        # 1. copy model (if necessary)        
        derivedModel = baseModel if (persistent and overwrite) else baseModel.copy(name = name)
        # 2. apply filter, i.e. condition
        equalConditions =  filter( lambda cond: "operator" in cond and cond["operator"] == "EQUALS", filters)
        pairs = list( map( lambda cond : ( cond["randVar"], cond["value"] ), equalConditions ) )
        derivedModel.condition(pairs)
        # 3. remove unneeded random variables
        derivedModel.marginalize(keep = randVars)
        # 4. store model in model base
        return derivedModel if not persistent else self._add(derivedModel, name)
 
    def _predict (self, aggrRandVars, model, filters=[], groupBy=[]):
        """ Runs a prediction query against the model base and returns its result
        by means of a data frame.

        The data frame contains exactly those columns/random variables which
        are specified in the aggrRandVars parameter. Its order is preserved.
        
        Args:
            aggrRandVars: 
        
        NOTE/TODO: SO FAR ONLY A VERY LIMITED VERSION IS IMPLEMENTED: 
        only a single aggrRandVar and no groupBys are allowed
        """
        
        # (1) derive the base model,
        # i.e. a model on all requested dimensions and measures, respecting filters 
        # var fields = query.fields();
       # usedRVs = set(map(lambda rv: rv["randVar"], aggrRandVars))
     #   base_model = _model()        
        # TODO: is there any filter that cannot be applied yet?
        
        
        # (2) derive a sub-model for each requested aggregation
        # i.e. remove all random variables of other measures which are not also a used as a dimension
        
        # (3) generate input for model aggregations,
        # i.e. a cross join of splits of all dimensions
        # note: filters on dimensions should already have been applied
                
        # (4) query models and fill result data frme
        # now question is:
        # * how to efficiently query the model?
        # * how can I vectorize it?
        # just start simple: don't do it vectorized! provide a vectorized
        # interface, but default to scalar implementation if the vectorized one
        # isn't implemented for a model
                
        # (5) return data frame
        
        # get list of RVs to group by
        
        """

        def crossJoin (dataframe, randVar):
            # extract split fct
            split = randVar
            series = split(randVar)            
            #todo: filter on series            
            # do cross join
            return crossjoin(dataframe, series)
                             
        # iteratively build input table           
        groupFrame = reduce(crossJoin, groupBy, pd.DataFrame())
        
        #2. setup input tuple, i.e. calculate the cross product of all dim.splitToValues()
        # pair-wise joins of dimension domains, i.e. create all combinations of dimension domain values
        '''let inputTable = dimensions.reduce(
        function (table, dim) {
            return _join(table, [dim.splitToValues()]);
            }, []);'''
        """
        
        if groupBy:
            raise NotImplementedError()
            # TODO make sure to implement non-aggregated randVars in the 
            # PREDICT-clause when implemening groupBy
        
        # assume: there should be aggregations attached to the randVars
        # assume: only 1 randVar, 
        if len(aggrRandVars) > 1:
            raise NotImplementedError()        
        aggrRandVar = aggrRandVars[0]
        
        # 1. derive required submodel
        predictionModel = self._model(randVars = [aggrRandVar["randVar"]], baseModel = model, name = _id_generator(), filters = filters, persistent = False)

        # 2. query it
        result = predictionModel.aggregate(aggrRandVar["aggregation"])

        # 3. convert to python scalar
        return result.item(0,0)
        
    def _show (self, query, show):
        """ Runs a 'SHOW'-query against the model base and returns its results. """
        if show == "HEADER": 
            model = self._extractFrom(query)
#            logger.debug("n = " + str(model._n))
#            logger.debug("S = \n" + str(model._S))
#            logger.debug("mu = \n" + str(model._mu))
#            logger.debug("fields = \n" + str(model.fields))
            return model.fields
        elif show == "MODELS":
            return list(map(lambda m: m.name, self.models.values()))
        
if __name__ == '__main__':
    import numpy as np
    mvg = gm.MultiVariateGaussianModel()
    mvg.fit()
    print(mvg._density(np.matrix('1 1 1 1').T))
    print(mvg._density(mvg._sample()))
    mb = ModelBase("mymodelbase")
    cc = mb.models['car_crashes']