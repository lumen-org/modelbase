# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

The modelbase module primarily provides the ModelBase class.
"""

import json
import logging
from functools import reduce
from pathlib import Path
import numpy

import models as gm

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


class NumpyCompliantJSONEncoder(json.JSONEncoder):
    """A JSON encoder that does *not* fail when serializing numpy.integer, numpy.floating or numpy.ndarray.
     Other than that it behaves like the default encoder.
    Credits to: http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    """

    # TODO: Do i really want that? what side effects does it have if I, all of the sudde, have numpy objects instead of normal integer and floats in my model??? it was never intended for it

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyCompliantJSONEncoder, self).default(obj)


def _json_dumps(*args, **kwargs):
    """Shortcut to serialize objects with my NumpyCompliantJSONEncoder"""
    return json.dumps(*args, **kwargs, cls=NumpyCompliantJSONEncoder)
    #return json.dumps(*args, **kwargs)


def PQL_parse_json(query):
    """ Parses a given PQL query and transforms it into a more readable and handy 
    format that nicely matches the interface of models.py.
    """

    def _predict(clause):
        #TODO refactor: this should be a method of AaggregationTuple: e.g. AggregationTuple.fromJSON
        #TODO refactor: density really is something else than an aggregation...
        def _aggrSplit(e):
            if isinstance(e, str):
                return e
            else:
                names = [e["name"]] if isinstance(e["name"], str) else e["name"]
                args = e["args"] if "args" in e else None
                yields = e["yields"] if "yields" in e else None
                try:
                    return gm.AggregationTuple(names, e["aggregation"], yields, args)
                except KeyError:
                    raise ValueError("unsupported aggregation method: " + str(e))

        return list(map(_aggrSplit, clause))

    def _where(clause):
        return [gm.ConditionTuple(e["name"], e["operator"], e["value"]) for e in clause]

    def _splitby(clause):
        def _mapSplit(e):
            args = e["args"] if "args" in e else None
            return gm.SplitTuple(e["name"], e["split"], args)

        return list(map(_mapSplit, clause))

    if "PREDICT" in query:
        query["PREDICT"] = _predict(query["PREDICT"])
    if "WHERE" in query:
        query["WHERE"] = _where(query["WHERE"])
    if "SPLIT BY" in query:
        query["SPLIT BY"] = _splitby(query["SPLIT BY"])
    # "SHOW", "AS", "FROM" and "MODEL" can stay as they are.
    return query


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

    def __init__(self, name, model_dir='data_models', load_all=True):
        """ Creates a new instance and loads models from some directory. """

        self.name = name
        self.models = {}  # models is a dictionary, using the name of a model as its key
        self.model_dir = model_dir

        # load some initial models to play with
        if load_all:
            self.load_all_models()

    def __str__(self):
        return " -- Model Base > " + self.name + " < -- \n" + \
               "contains " + str(len(self.models)) + " models, as follows:\n\n" + \
               reduce(lambda p, m: p + str(m) + "\n\n", self.models.values(), "")

    def load_all_models(self, directory=None, ext='.mdl'):
        """Loads all models from the given directory. Each model is expected to be saved in its own file. Only files
         that end on like given by parameter ext are considered.
         If there is any file that matches the naming convention but doesn't contain a model a warning is issued and
         it is skipped.

         Args:
             directory: directory to store the models in. Defaults to the set directory of the model base.
             ext: file extension to use when loading models.
         """
        if directory is None:
            directory = self.model_dir

        # iterate over matching files in directory (including any subdirectories)
        filenames = Path(directory).glob('**/' + '*' + ext)
        for file in filenames:
            # try loading the model
            try:
                model = gm.Model.load(str(file))
                self.add(model)
            except TypeError as err:
                print(str(err))
                logger.warn('file "' + str(file) + '" matches the naming pattern but does not contain a model instance')

    def save_all_models(self, directory=None, ext='.mdl'):
        """Saves all models currently in the model base in given directory using the naming convention:
         <model-name>.<ext>

         Args:
             directory: directory to store the models in. Defaults to the set directory of the model base.
             ext: file extension to use when saving models.
         """
        if directory is None:
            directory = self.model_dir
        dir_ = Path(directory)
        for key, model in self.models.items():
            filepath = dir_.joinpath(model.name + ext)
            gm.Model.save(model, str(filepath))

    def add(self, model, name=None):
        """ Adds a model to the model base using the given name or the models name. """
        if name in self.models:
            logger.warn('Overwriting existing model in model base: ' + name)
        if name is None:
            name = model.name
        self.models[name] = model
        return None

    def drop(self, name):
        """ Drops a model from the model base and returns the dropped model."""
        model = self.models[name]
        del self.models[name]
        return model

    def drop_all(self):
        """ Drops all models of this modelbase."""
        names = [name for name in self.models]  # create copy of keys in model
        for name in names:
            self.drop(name)

    def get(self, name):
        """Gets a model from the modelbase by name and returns it."""
        return self.models[name]

    def list_models(self):
        return list(self.models.keys())

    def execute(self, query):
        """ Executes the given PQL query and returns the result as JSON (or None).

        Args:
            query: A word of the PQL language, either as JSON or as a string.

        Returns:
            The result of the query as a JSON-string, i.e. 
            '''json.loads(result)''' works just fine on it.

        Raises:
            QuerySyntaxError: If there is some syntax error.
            QueryValueError: If there is some problem with a value of the query.
        """

        # parse query
        # turn to JSON if not already JSON
        if isinstance(query, str):
            query = json.loads(query)
        query = PQL_parse_json(query)

        # basic syntax and semantics checking of the given query is done in the _extract* methods
        if 'MODEL' in query:
            base = self._extractFrom(query)
            # maybe copy
            derived_model = base if base.name == query["AS"] else base.copy(query["AS"])
            # derive submodel
            derived_model.model(
                model=self._extractModel(query),
                where=self._extractWhere(query))
            # add to modelbase
            self.add(derived_model, query["AS"])
            # return header
            return _json_dumps({"name": derived_model.name, "fields": derived_model.json_fields()})

        elif 'PREDICT' in query:
            base = self._extractFrom(query)
            resultframe = base.predict(
                predict=self._extractPredict(query),
                where=self._extractWhere(query),
                splitby=self._extractSplitBy(query))
            return _json_dumps({"header": resultframe.columns.tolist(),
                               "data": resultframe.to_csv(index=False, header=False)})

        elif 'DROP' in query:
            self.drop(name=query['DROP'])
            return _json_dumps({})

        elif 'SHOW' in query:
            show = self._extractShow(query)
            if show == "HEADER":
                model = self._extractFrom(query)
                result = {"name": model.name, "fields": model.json_fields()}
            elif show == "MODELS":
                result = self.list_models()
            else:
                raise ValueError("invalid value given in SHOW-clause: " + str(show))
            return _json_dumps(result)

    ### _extract* functions are helpers to extract a certain part of a PQL query
    #   and do some basic syntax and semantic checks

    def _extractFrom(self, query):
        """ Returns the model that the value of the "FROM"-statement of query
        refers to. """
        if 'FROM' not in query:
            raise QuerySyntaxError("'FROM'-statement missing")
        modelName = query['FROM']
        if modelName not in self.models:
            raise QueryValueError("The specified model does not exist: " + modelName)
        return self.models[modelName]

    def _extractShow(self, query):
        """ Extracts the value of the "SHOW"-statement from query."""
        if 'SHOW' not in query:
            raise QuerySyntaxError("'SHOW'-statement missing")
        what = query['SHOW']
        if what not in ["HEADER", "MODELS"]:
            raise QueryValueError("Invalid value of SHOW-statement: " + what)
        return what

    def _extractSplitBy(self, query, required=False):
        """ Extracts from query the list of groupings. The order is preserved.

        Args:
            query: The query.
            required: Optional flag. Set to True to allow for a missing
                "SPLIT BY" statement in the query.

        Returns:
            The list of fields to group by. The Fields are dicts that are
            identical to the JSON in the query, i.e. they have two keys:
            'name' which holds the name of the random variable, and
            'split', the split method to use.
        """
        if required and 'SPLIT BY' not in query:
            raise QuerySyntaxError("'SPLIT BY'-statement missing")
        elif not required and 'SPLIT BY' not in query:
            return []
        return query['SPLIT BY']

    def _extractModel(self, query):
        """ Extracts the names of the random variables to model and returns it
        as a list of strings. The order is preserved.

        Note that it returns only strings, not actual Field objects.

        TODO: internally this function refers to self.models[query['FROM']].
            Remove this dependency
        """
        if 'MODEL' not in query:
            raise QuerySyntaxError("'MODEL'-statement missing")
        if query['MODEL'] == '*':
            return self.models[query['FROM']].names
        else:
            return query['MODEL']

    def _extractPredict(self, query):
        """ Extracts from query the fields to predict and returns them in a
        list. The order is preserved.

        The Fields are dicts that are identical to the JSON in the query.
        """
        if 'PREDICT' not in query:
            raise QuerySyntaxError("'PREDICT'-statement missing")
        return query['PREDICT']

    def _extractAs(self, query):
        """ Extracts from query the name under which the derived model is to be
        stored and returns it.
        """
        if 'AS' not in query:
            raise QuerySyntaxError("'AS'-statement missing")
        return query['AS']

    def _extractWhere(self, query, required=False):
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


if __name__ == '__main__':
    mb = ModelBase("mymb")
    cc = mb.models['car_crashes']
    foo = cc.copy().model(["total", "alcohol"], [gm.ConditionTuple("alcohol", "equals", 10)])
    print(str(foo))
