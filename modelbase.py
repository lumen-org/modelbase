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
import splitter as sp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QuerySyntaxError(Exception):
    meaning = """This error indicates that a PQL query was incomplete and\
 hence could not be executed"""
    
    def __init__(self, message="", value=None):
        self.value = value
        self.message = message
        
    def __repr__(self):
        return repr(self.value)
        

class QueryValueError(Exception):    
    meaning = """This error indicates that a PQL query contains a value that is\
 semantically invalid, such as referring to a model that does not exist."""

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
    
    It is, in that sense, the PQL-interface for models. It is build against the
    'raw' python interface for models, provided by the Model class.
    
    A Modelbase provides only one 'public' method:
        ModelBase.execute: It executes a given PQL query. All input and output 
            is provided as JSON objects, or a single string / number if 
            applicable.
       
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

### _extract* functions are helpers to extract a certain part of a PQL query
#   and do some basic syntax and semantic checks
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
        """ Extracts from query the list of groupings. The order is preserved.
        
        Args:
            query: The query.
            required: Optional flag. Set to True to allow for a missing 
                "GROUP BY" statement in the query.
            
        Returns:
            The list of fields to group by. The Fields are dicts that are 
            identical to the JSON in the query, i.e. they have two keys: 
            'name' which holds the name of the random variable, and
            'split', the split method to use.
        """
        if required and 'GROUP BY' not in query:
            raise QuerySyntaxError("'GROUP BY'-statement missing")
        elif not required and 'GROUP BY' not in query:
            return []            
        #return list( map( lambda v: v['name'], query['GROUP BY'] ) )
        return query['GROUP BY']
            
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
            return list( map( lambda v: v['name'], query['MODEL'] ) )
        
    def _extractPredict (self, query): 
        """ Extracts from query the fields to predict and returns them in a
        list. The order is preserved.
        
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
            derived_model = self._model( randVars = self._extractModel(query),
                        baseModel = self._extractFrom(query),
                        name = self._extractAs(query), 
                        filters = self._extractWhere(query) )
            return self._show(query=derived_model, show="HEADER")

        elif 'PREDICT' in query:
            result = self._predict( predict = self._extractPredict(query),
                    model = self._extractFrom(query),
                    filters = self._extractWhere(query),
                    group_by = self._extractGroupBy (query) )
            return result
        
        elif 'DROP' in query:
            self._drop(name = query['DROP'])
            return None
            
        elif 'SHOW' in query:
            return self._show( query = query, show = self._extractShow(query))
       
    def _add  (self, model, name):
        """ Adds a model to the model base using the given name. """
        if name in self.models:
            logger.warn('Overwriting existing model in model base: ' + name)
        self.models[name] = model
        return None
    
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
        pairs = list( map( lambda cond : ( cond['name'], cond["value"] ), equalConditions ) )
        derivedModel.condition(pairs)
        # 3. remove unneeded random variables
        derivedModel.marginalize(keep = randVars)
        # 4. store model in model base
        return derivedModel if not persistent else self._add(derivedModel, name)
 
    def _predict (self, predict, model, filters=[], group_by=[]):
        """ Runs a prediction query against the model base and returns its result
        by means of a data frame.

        The data frame contains exactly those columns/random variables which
        are specified in the aggrRandVars parameter. Its order is preserved.
        
        Args:
            predict: A list of the random variables to predict, i.e. to include
                in the result table. The random variables are either be 
                aggregations of one or more random variables, or dimensions, i.e.
                random variables that are split by. Hence, each RV is a dict with 
                (at least) one key:
                    'name': the name of the RV. 
                Furthemore, if existent, another key is respected:
                    'aggregation': the aggregation to use for the RV
            model: The model to predict from.
            filters: A list of filters to use. 
            groupBy: A list of random variables to group by. Each RV is a dict 
                with (at least) two keys:
                    'name': the name of the RV, and
                    'split': the split method to use.
        
        NOTE/TODO: SO FAR ONLY A VERY LIMITED VERSION IS IMPLEMENTED: 
        only a single aggrRandVar and no groupBys are allowed
        """       
        # (1) derive the base model,
        # i.e. a model on all requested dimensions and measures, respecting filters 
        # TODO: is there any filter that cannot be applied yet?        
        
        # derive the list of dimensions to split by
        split_names = list(map(lambda f: f['name'], group_by))
        # derive the list of aggregations and dimensions to include in result table        
        aggrs, aggr_names, dim_names, predict_names = [], [], [], []
        for f in predict:
            name = f['name']
            predict_names.append(name)
            if 'aggregation' in f:
                aggr_names.append(name)
                aggrs.append(f)
            else:
                dim_names.append(name)        
        # from that derive the set of (names of) random variables that are to be kept for the base model
        base_names = list(set(split_names) | set(aggr_names) | set(dim_names))
        # now get the base model    
        base_model = self._model(randVars = base_names, 
                            baseModel = model,
                            name = '__' + model.name + '_base',
                            filters = filters,
                            persistent = True) #TODO: REMOVE THAT LATER!
        
        # (2) derive a sub-model for each requested aggregation
        # i.e. remove all random variables of other measures which are not also a used for splitting
        # or equivalently: keep all random variables of dimensions, plus the once for the current aggregation
        dim_names_unique = set(split_names) # WHICH ARE NOT ALSO USED AS A DIMENSION
        i = 0
        def derive_aggregation_model (aggr_name):
            nonlocal i
            model = self._model(randVars = list(dim_names_unique | set([aggr_name])),
                                baseModel = base_model,
                                name = base_model.name + "_" + aggr_name + str(i),
                                persistent = True) # TODO: REMOVE THAT LATER
            i+=1
            return model
        # TODO: use a different naming scheme later, e.g.: name = _id_generator(),                
        aggr_models = list(map(derive_aggregation_model, aggr_names))

        # TODO: derive model for density
        # TODO: is density really just another aggregation?        
        
        # (3) generate input for model aggregations,
        # i.e. a cross join of splits of all dimensions
        # note: filters on dimensions should already have been applied
        def _get_group_frame (randVar):
            name = randVar['name']
            domain = base_model._byName(name)["domain"]
            domain = sp.NumericDomain(domain[0], domain[1])
            splitFct = sp.splitter[randVar["split"]]
            frame = pd.DataFrame( splitFct(domain, 3), columns = [name])
            frame['__crossIdx__'] = 0 # we need that index to crossjoin later
            return frame
        def _crossjoin (df1, df2):
            return pd.merge(df1, df2, on='__crossIdx__', copy=False)
        group_frames = map(_get_group_frame, group_by)
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
            aggr_method = aggr["aggregation"]
            aggr_name = aggr['name']
            for row in input_frame.iterrows():
                # derive model for these specific conditions...
                pairs = zip(split_names, row[1])
                mymodel = aggr_model.copy().condition(pairs).marginalize(keep = [aggr_name])
                # now do the aggregation
                # TODO: in the future, there may be multidimensional aggregations
                # for now, it's just 1d --> [0]
                res = mymodel.aggregate(aggr_method)[0]                
                aggr_results.append(res)
            series = pd.Series(aggr_results, name=aggr_name)
            result_list.append(series)
               
        # (5) filter on aggregations?
        # TODO?
        
        # (6) collect all into one data frame
        return_frame = pd.concat(result_list, axis=1)
                
        # (7) return correctly ordered frame that only contain requested variables
        # TOOD: this is buggy for the case of a field being returned multiple times...
        return return_frame[predict_names].to_json()

        
    def _show (self, query, show):
        """ Runs a 'SHOW'-query against the model base and returns its results. """
        if show == "HEADER": 
            model = self._extractFrom(query)
#            logger.debug("n = " + str(model._n))
#            logger.debug("S = \n" + str(model._S))
#            logger.debug("mu = \n" + str(model._mu))
#            logger.debug("fields = \n" + str(model.fields))
            return str(model.fields)
        elif show == "MODELS":
            return str(list(map(lambda m: m.name, self.models.values())))
        
if __name__ == '__main__':
    import numpy as np
    mvg = gm.MultiVariateGaussianModel()
    mvg.fit()
    print(mvg._density(np.matrix('1 1 1 1').T))
    print(mvg._density(mvg._sample()))
    mb = ModelBase("mymodelbase")
    cc = mb.models['car_crashes']