"""
@author: Philipp Lucas

The modelbase module primarly provides the ModelBase class.
"""

import logging
import string
import random
import models as gm
from functools import reduce
import seaborn.apionly as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QuerySyntaxError(Exception):
    meaning = '''This error indicates that a PQL query was incomplete and hence could not be executed'''
    
    def __init__(self, message="", value=None):
        self.value = value
        self.message = message
        
    def __repr__(self):
        return repr(self.value)
        

class QueryValueError(Exception):    
    meaning = '''This error indicates that a PQL query contains a value that is semantically invalid, such as referring to a model that does not exist.'''

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
    '''loads the iris data set and returns a MultiVariateGaussianModel of it'''    
    # load data set as pandas DataFrame
    data = sns.load_dataset('iris')
    # train model on continuous part of the data
    model = gm.MultiVariateGaussianModel('iris', data.iloc[:, 0:-1])
    model.fit()        
    return model
    
def _loadCarCrashModel ():     
    '''loads the car crash data set and returns a MultiVariateGaussianModel of it'''
    data = sns.load_dataset('car_crashes')
    model = gm.MultiVariateGaussianModel('car_crashes', data.iloc[:, 0:-1])
    model.fit()
    return model

def _id_generator(size=15, chars=string.ascii_letters + string.digits, prefix='__'):
    '''Returns a prefixed, random string of letters and digits '''
    return prefix + ''.join(random.choice(chars) for _ in range(size))    

class ModelBase:
    '''A ModelBase is the analogon of a DataBase(-Management System) for models: it holds models and allows PQL queries against them.
    
       A Modelbase provides only one 'public' method:
         * *var* ModelBase.execute: It executes a given PQL query. All input
         and output is provided as JSON objects, or simple strings / numbers 
         if applicable.
       
       Furthermore it provides to 'public' attributes: 
       
         * *var* ModelBase.name: the id/name of this modelbase, and
         * *var* ModelBase.models: a dictionary of all models in the modelbase. 
         Each model must have a unique string as its name, and this name is used
         as a key in the dictionary.
    '''
    def __init__ (self, name):
        '''creats a new instance and loads some default models '''
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
        '''returns the model that the value of the "FROM"-statement of query refers to and checks basic syntax and semantics'''
        if 'FROM' not in query:
            raise QuerySyntaxError("'FROM'-statement missing")
        modelName = query['FROM']
        if modelName not in self.models:
            raise QueryValueError("The specified model does not exist: " + modelName)            
        return self.models[modelName]
    
    def _extractShow (self, query):
        '''extracts the value of the "SHOW"-statement from query and checks basic syntax and semantics'''        
        if 'SHOW' not in query:
            raise QuerySyntaxError("'SHOW'-statement missing")
        what = query['SHOW']
        if what not in ["HEADER", "MODELS"]:
            raise QueryValueError("Invalid value of SHOW-statement: " + what)
        return what
        
    def _extractGroupBy (self, query, required=False):        
        if required and 'GROUP BY' not in query:
            raise QuerySyntaxError("'GROUP BY'-statement missing")
        elif not required and 'GROUP BY' not in query:
            return []            
        return list( map( lambda v: v['randVar'], query['GROUP BY'] ) )
            
    def _extractModel (self, query):
        if 'MODEL' not in query:
            raise QuerySyntaxError("'MODEL'-statement missing")            
        if query['MODEL'] == '*':
            return list( map( lambda field: field["name"], self.models[query['FROM']].fields ) )
        else:            
            return list( map( lambda v: v['randVar'], query['MODEL'] ) )
        
    def _extractPredict (self, query): 
        if 'PREDICT' not in query:
            raise QuerySyntaxError("'PREDICT'-statement missing")
        # return in one list                    
        return query['PREDICT']        
        
    def _extractAs (self, query):        
        if 'AS' not in query:
            raise QuerySyntaxError("'AS'-statement missing")
        return query['AS']
        
    def _extractWhere (self, query, required=False):
        if required and 'WHERE' not in query:
            raise QuerySyntaxError("'WHERE'-statement missing")
        elif not required and 'WHERE' not in query:
            return []            
        else:
            return query['WHERE']
    
    def execute (self, query):
        '''executes the given PQL query and returns a status code and the result (or None)
        
        *var* query must be a word of the PQL language.'''
        
        # basic syntax and semantics checking of the given query is done in the _extract* methods
        if 'MODEL' in query:            
            self._model( randVars = self._extractModel(query),
                        baseModel = self._extractFrom(query),
                        name = self._extractAs(query), 
                        filters = self._extractWhere(query) )
            return ReturnCode["SUCCESS"], None

        elif 'PREDICT' in query:
            result = self._predict( aggrRandVars = self._extractPredict(query),
                    model = self._extractFrom(query),
                    filters = self._extractWhere(query),
                    groupBy = self._extractGroupBy (query) )
            return ReturnCode["SUCCESS"], result
        
        elif 'DROP' in query:
            self._drop(name = query['DROP'])
            return ReturnCode["SUCCESS"], None
            
        elif 'SHOW' in query:
            header = self._show( query = query, show = self._extractShow(query))
            return ReturnCode["SUCCESS"], header            
       
    def _add  (self, model, name):
        ''' adds a model to the model base'''
        if name in self.models:
            logger.warn('Overwriting existing model in model base: ' + name)
        self.models[name] = model
        return model
    
    def _drop (self, name):
        ''' drops a model from the model base and returns the dropped model.
            Returns None if there is no model with name *var* name. '''
        if name in self.models:
            model = self.models[name]
            del self.models[name]
            return model
        else:
            return None
        
    def _model (self, randVars, baseModel, name, filters=[]):
        ''' runs a 'model' query against the model base and returns its result'''
        # 1. copy model (if necessary)
        derivedModel = baseModel if baseModel.name == name else baseModel.copy(name = name)
        # 2. apply filter, i.e. condition
        equalConditions =  filter( lambda cond: "operator" in cond and cond["operator"] == "EQUALS", filters)
        pairs = list( map( lambda cond : ( cond["randVar"], cond["value"] ), equalConditions ) )
        derivedModel.condition(pairs)
        # 3. remove unneeded random variables
        derivedModel.marginalize(keep = randVars)        
        # 4. store model in model base
        return self._add(derivedModel, name)
        
    def _predict (self, aggrRandVars, model, filters=[], groupBy=[]):
        '''runs a prediction query against the model base and returns its result
        
        NOTE/TODO: SO FAR ONLY A VERY LIMITED VERSION IS IMPLEMENTED: only a single aggrRandVar and no groupBys are allowed'''
        if groupBy:
            raise NotImplementedError()
            # make sure to implement non-aggregated randVars in the PREDICT-clause when implemening groupBy
        # assume: there should be aggregations attached to the randVars
        # assume: only 1 randVar, 
        if len(aggrRandVars) > 1:
            raise NotImplementedError()        
        aggrRandVar = aggrRandVars[0]
        # 1. derive required submodel
        predictionModel = self._model(randVars = [aggrRandVar["randVar"]], baseModel = model, name = _id_generator(), filters = filters)
        # 2. query it
        result = predictionModel.aggregate(aggrRandVar["aggregation"])
        # 3. convert to python scalar
        return result.item(0,0)
        
    def _show (self, query, show):
        '''runs a 'SHOW'-query against the model base and returns its results'''
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