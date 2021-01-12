# Copyright (c) 2017-2018 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

The modelbase module primarily provides the ModelBase class.
"""

import json
import logging
import os
import pathlib
import time
from functools import reduce

import dill
import numpy as np

from ..core import base, model_watchdog, models_predict, models as md
from ..eval import posterior_predictive_checking as ppc
from ..utils import model_fitting

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class QueryIncompleteError(Exception):
    meaning = """This error indicates that a PQL query was incomplete, i.e. it could not be successfully interpreted because required information was missing."""

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

    # TODO: Do i really want that? what side effects does it have if I, all of
    # the sudde, have numpy objects instead of normal integer and floats in my
    # model??? it was never intended for it

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyCompliantJSONEncoder, self).default(obj)


def _json_dumps(*args, **kwargs):
    """Shortcut to serialize objects with my NumpyCompliantJSONEncoder"""
    return json.dumps(*args, **kwargs, cls=NumpyCompliantJSONEncoder)
    # return json.dumps(*args, **kwargs)


def PQL_parse_json(query):
    """ Parses a given PQL query and transforms it into a more readable and handy
    format that nicely matches the interface of models.py.
    """

    def _predict(clause):
        # TODO refactor: this should be a method of AaggregationTuple: e.g. AggregationTuple.fromJSON
        # TODO refactor: density really is something else than an
        # aggregation...
        def _aggrSplit(e):
            if isinstance(e, str):
                return e
            else:
                names = [e["name"]] if isinstance(
                    e["name"], str) else e["name"]
                args = e["args"] if "args" in e else None
                yields = e["yields"] if "yields" in e else None
                try:
                    return base.AggregationTuple(
                        names, e["aggregation"], yields, args)
                except KeyError:
                    raise ValueError(
                        "unsupported aggregation method: " + str(e))

        return list(map(_aggrSplit, clause))

    def _where(clause):
        return [
            base.Condition(
                e["name"],
                e["operator"],
                e["value"]) for e in clause]

    def _splitby(clause):
        def _mapSplit(e):
            args = e["args"] if "args" in e else None
            return base.SplitTuple(e["name"], e["split"], args)

        return list(map(_mapSplit, clause))

    if "PREDICT" in query:
        query["PREDICT"] = _predict(query["PREDICT"])
    if "WHERE" in query:
        query["WHERE"] = _where(query["WHERE"])
    if "SPLIT BY" in query:
        query["SPLIT BY"] = _splitby(query["SPLIT BY"])
    # "SHOW", "AS", "FROM" and "MODEL" can stay as they are.
    return query


def _check_if_dir_exists(model_dir):
    if not pathlib.Path(model_dir).is_dir():
        raise OSError('Path is not a directory: {}'.format(str(model_dir)))


class ModelBase:
    """A ModelBase is the analog of a DataBase Management System for
    models: it holds models and allows queries against them.

    Instead of SQL it uses PQL. It is, in that sense, the PQL-interface
    for models. It is built against the 'raw' python interface for models,
    provided by the Model class.

    Query Execution:
        ModelBase.execute: It executes a given PQL query. All input and output
            is provided as JSON objects, or a single string / number if
            applicable.

    Attributes:
        name (str): the id/name of this modelbase.
        models: a dictionary of all models in the modelbase. Each model must have a unique string
            as its name, and this name is used as a key in the dictionary.
        model_dir (str): The path to the directory where the models are managed.
        settings: A dictionary with various settings:
            .float_format : The float format used to encode floats in a result. Defaults to '%.5f'
        cache (cache.BaseCache): a cache to store models and queries for reuse.
        log_queries (bool): Log executed queries to a file
        query_log_path (str): Path to the query log file

    Args:
        name (str): the id/name of this modelbase.
        model_dir (str): The path to the directory where the models are managed.
        load_all (bool): Load all models in the model_dir at startup if True.
        cache (cache.BaseCache): A cache to store models and queries for reuse. Optional.
            Pass None to disable caching.
        watchdog (bool): Observe model_dir for changes.
        log_queries (bool): Log executed queries to a file
        query_log_path (str): Path to the query log file
    """

    def __init__(
            self,
            name,
            model_dir='data_models',
            auto_load_models={},
            load_all=True,
            cache=None,
            watchdog=True,
            log_queries=False,
            query_log_path='./'):
        """ Creates a new instance and loads models from some directory. """

        _check_if_dir_exists(model_dir)

        self.name = name
        self.models = {}
        self._modelname_by_filename = {}
        self.model_dir = model_dir
        self.cache = cache
        self.log_queries = log_queries
        self.query_log_path = os.path.abspath(query_log_path)
        self.settings = {
            'float_format': '%.8f',
        }

        # load some initial models to play with
        if load_all:
            logger.info("Loading models from directory '" + model_dir + "'")
            loaded_models = self.load_all_models()
            if len(loaded_models) == 0:
                logger.warning(
                    "I did not load ANY model. Make sure the provided model directory is correct! "
                    "I continue anyway.")
            else:
                logger.info("Successfully loaded " +
                            str(len(loaded_models)) +
                            " models into the modelbase: ")
                logger.info(str([model[0] for model in loaded_models]))

        self.model_watch_observer = None
        if watchdog:
            # init watchdog who oversees a given folder for new models
            self.model_watch_observer = model_watchdog.ModelWatchObserver()
            try:
                self.model_watch_observer.init_watchdog(
                    self, self.model_dir, **auto_load_models)
                logger.info(
                    "Files under {} are watched for changes".format(
                        self.model_dir))
            except Exception as err:
                logger.exception("Watchdog failed!")
                logger.exception(err)

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

         Returns:
             A list containing pairs of <name-of-loaded-model, file-name>
         """
        if directory is None:
            directory = self.model_dir

        _check_if_dir_exists(directory)

        # iterate over matching files in directory (including any subdirectories)
        loaded_models = []
        filepaths = pathlib.Path(directory).glob('**/' + '*' + ext)
        for filepath in filepaths:
            filename = filepath.name
            filepath = str(filepath)
            logger.debug("loading model from file: " + filepath)
            # try loading the model
            try:
                model = md.Model.load(filepath)
            except TypeError as err:
                logger.warning(
                    'file "' +
                    filepath +
                    '" matches the naming pattern but does not contain a model instance. '
                    'I ignored that file')
            else:

                self.add(model, filename=filename)
                loaded_models.append((model.name, filepath))
        return loaded_models

    def save_all_models(self, directory=None, ext='.mdl'):
        """Saves all models currently in the model base in given directory using the naming convention:
         <model-name>.<ext>

         Args:
             directory: directory to store the models in. Defaults to the set directory of the model base.
             ext: file extension to use when saving models.
         """
        if directory is None:
            directory = self.model_dir

        if not os.path.exists(directory):
            os.makedirs(directory)

        for key, model in self.models.items():
            md.Model.save_static(model, directory)

    def add(self, model, name=None, filename=None):
        """ Adds a model to the model base using the given name or the models name. """
        if name in self.models:
            logger.warning('Overwriting existing model in model base: ' + name)
        if name is None:
            name = model.name
        if filename is not None:
            self._modelname_by_filename[filename] = model.name
        self.models[name] = model
        return model

    def drop(self, name):
        """ Drops a model from the model base and returns the dropped model."""
        model = self.models[name]
        del self.models[name]

        for key, modelname in self._modelname_by_filename.items():
            if modelname == name:
                filename = key
                break
        del self._modelname_by_filename[filename]

    def drop_all(self):
        """ Drops all models of this modelbase."""
        names = [name for name in self.models]  # create copy of keys in model
        for name in names:
            self.drop(name)

    def get(self, name=None, filename=None):
        """Gets a model from the modelbase by name or filename and returns it or None if such a
        model does not exist.

        Arguments:
            name: string, optional.
                Name of the model to retrieve.
            filename: string, optional.
                The filename of the model to retrieve. Do not pass a full path, just the filename.
        Returns: Model
        """
        if name is None and filename is None:
            raise ValueError("Neither name nor filename given.")

        if filename is not None:
            name = self._modelname_by_filename.get(filename, None)
        return self.models.get(name, None)

    def list_models(self):
        return list(self.models.keys())

    def execute(self, query):
        """ Executes the given PQL query and returns the result as JSON (or None).

        Note that queries may be cached, if caching is enabled. See also Modelbase.

        Args:
            query: A word of the PQL language, either as JSON or as a string.

        Returns:
            The result of the query as a JSON-string, i.e.
            '''json.loads(result)''' works just fine on it.

        Raises:
            QuerySyntaxError: If there is some syntax error.
            QueryValueError: If there is some problem with a value of the query.
        """

        if self.log_queries:
            query_log_name = self.query_log_path + "/interaction" + \
                             time.strftime("%b:%d:%Y_%H", time.gmtime(time.time())) + ".log"
            if "SHOW" not in query.keys():
                with open(query_log_name, "a") as f:
                    f.write(json.dumps(query) + '\n')
                    logger.info(json.dumps(query))

        # parse query
        # turn to JSON if not already JSON
        if isinstance(query, str):
            query = json.loads(query)
        query = PQL_parse_json(query)

        # basic syntax and semantics checking of the given query is done in the
        # _extract* methods
        if 'MODEL' in query:
            base = self._extractFrom(query)
            key = self.model_key(query)

            derived_model = None
            if self.cache is not None:
                derived_model = self.cache.get(key)

            if derived_model is None:
                # maybe copy
                derived_model = base if base.name == query["AS"] else base.copy(
                    query["AS"])
                # derive submodel
                derived_model.model(
                    model=self._extractVariables(query),
                    where=self._extractWhere(query),
                    default_values=self._extractDefaultValue(query),
                    default_subsets=self._extractDefaultSubset(query),
                    hide=self._extractHide(query)),
                if self.cache is not None:
                    self.cache.set(key, derived_model)
            else:
                derived_model.set_default_value(
                    self._extractDefaultValue(query)) .set_default_subset(
                    self._extractDefaultSubset(query)) .hide(
                    self._extractHide(query))

            # add to modelbase
            self.add(derived_model, query["AS"])
            # return header
            return _json_dumps({"name": derived_model.name,
                                "fields": derived_model.json_fields()})

        elif 'SELECT' in query:
            base = self._extractFrom(query)
            resultframe = base.select(
                what=self._extractSelect(query),
                where=self._extractWhere(query),
                **self._extractOpts(query)
            )

            return _json_dumps({"header": resultframe.columns.tolist(
            ), "data": resultframe.to_csv(index=False, header=False)})

        elif 'PREDICT' in query:
            key = self.predict_key(query)
            resultframe = None
            if self.cache is not None:
                resultframe = self.cache.get(key)

            if resultframe is None:
                # Extract multiple
                base = self._extractFrom(query)
                predict_stmnt = self._extractPredict(query)
                where_stmnt = self._extractWhere(query)
                splitby_stmnt = self._extractSplitBy(query)

                resultframe = base.predict(
                    predict=predict_stmnt,
                    where=where_stmnt,
                    splitby=splitby_stmnt,
                    **self._extractOpts(query)
                )
                if self.cache is not None:
                    self.cache.set(key, resultframe)

            # TODO: is this working?
            if 'DIFFERENCE_TO' in query:
                base2 = self._extractDifferenceTo(query)
                resultframe2 = base2.predict(
                    predict=predict_stmnt,
                    where=where_stmnt,
                    splitby=splitby_stmnt
                )
                assert (resultframe.shape == resultframe2.shape)
                aggr_idx = [i for i, o in enumerate(predict_stmnt)
                            if models_predict.type_of_clause(o) != 'split']

                # calculate the diff only on the _predicted_ variables
                if len(aggr_idx) > 0:
                    resultframe.iloc[:,
                                     aggr_idx] = resultframe.iloc[:,
                                                                  aggr_idx] - resultframe2.iloc[:,
                                                                                                aggr_idx]

            resultframe_csv = resultframe.to_csv(index=False,
                                                 header=False,
                                                 float_format=self.settings['float_format'])

            return _json_dumps({"header": resultframe.columns.tolist(), "data": resultframe_csv})

        elif 'PPC' in query:  # posterior predictive checks
            var_names = self._extractVariables(query,keyword='PPC')
            base_model = self._extractFrom(query)
            opts = self._extractOpts(query)

            if not 'TEST_QUANTITY' in opts:
                raise ValueError("missing parameter 'TEST_QUANTITY'")
            test_quantity = opts['TEST_QUANTITY']
            if not test_quantity in ppc.TestQuantities:
                raise ValueError("invalid value for parameter 'TEST_QUANTITY': '{}'".format(test_quantity))
            test_quantity_fct = ppc.TestQuantities[test_quantity]

            # TODO: issue #XX: it is a questionable design decision: is it a good idea to marginalize a model
            #  just to create marginal samples? e.g. for cg models it should be faster to not marginalize but simply
            #  throw not needed attributes from the samples.
            marginal_model = base_model.copy().marginalize(keep=var_names)
            res = ppc.posterior_predictive_check(marginal_model, test_quantity_fct,
                                                 round(opts.get('k', None)), round(opts.get('n', None)))
            res_dict = {
                'header': var_names,
                'reference': res[0],
                'test': res[1]
            }
            return _json_dumps(res_dict)

        elif 'DROP' in query:
            self.drop(name=query['DROP'])
            return _json_dumps({})

        elif 'SHOW' in query:
            show = self._extractShow(query)
            if show == "HEADER":
                model = self._extractFrom(query)
                result = model.as_json()
            elif show == "MODELS":
                result = {'models': self.list_models()}
            else:
                raise ValueError(
                    "invalid value given in SHOW-clause: " + str(show))
            return _json_dumps(result)

        elif 'RELOAD' in query:
            what = self._extractReload(query)
            if what == '*':
                loaded_models = self.load_all_models()
                return _json_dumps({'STATUS': 'success', 'models': [
                                   model[0] for model in loaded_models]})
            else:
                raise ValueError("not implemented")

        elif 'PCI_GRAPH.GET' in query:
            model = self._extractFrom(query)
            #TODO: why is this disabled? fix it?
            #graph = pci_graph.to_json(model.pci_graph) if model.pci_graph else False
            graph = False
            return _json_dumps({
                'model': model.name,
                'graph': graph
            })

        elif 'PP_GRAPH.GET' in query:
            model = self._extractFrom(query)
            pp_graph = model.probabilistic_program_graph
            graph = pp_graph if pp_graph else False
            return _json_dumps({
                'model': model.name,
                'graph': graph
            })

        elif 'CONFIGURE' in query:
            model = self._extractModelByStatement(query, 'CONFIGURE')
            config = self._extractDictByKeyword(query, 'WITH')
            model.set_configuration(config)
            return _json_dumps({
                'model': model.name,
                'status': 'configuration changed'
            })

        else:
            raise QueryIncompleteError(
                "Missing Statement-Type (e.g. DROP, PREDICT, SELECT)")

    def upload_files(self, models):
        """
        saves given dill objects into the model-dir folder if they do not exist

        :param models: list of dumped models
        :return: "OK" if worked, else Error
        """
        model_list_saved = []
        model_list_existing = []
        for model in models:
            try:
                model = dill.loads(model)
                if isinstance(
                        model,
                        md.Model) and model.name not in self.models:
                    model.save(self.model_dir)
                    model_list_saved.append(model.name)
                else:
                    model_list_existing.append(model.name)
            except Exception as e:
                logger.exception(e)
                return "Error with pickle"

        logger.info("Models saved: {}".format(model_list_saved))
        logger.info("Models ignored: {}".format(model_list_existing))
        return "OK"

    # _extract* functions are helpers to extract a certain part of a PQL query
    #   and do some basic syntax and semantic checks

    def _extractModelByStatement(self, query, keyword):
        """ Returns the model that the value of the <keyword<-statement of query
        refers to. """
        if keyword not in query:
            raise QuerySyntaxError("{}-statement missing".format(keyword))
        model_name = query[keyword]

        if model_name not in self.models:
            self._automatically_create_model(model_name, self._extractOpts(query))

        return self.models[model_name]

    def _extractFrom(self, query):
        """ Returns the model that the value of the "FROM"-statement of query
        refers to. """
        return self._extractModelByStatement(query, 'FROM')

    def _extractDifferenceTo(self, query):
        """ Returns the model that the value of the "DIFFERENCE_TO"-statement of query
        refers to. """
        return self._extractModelByStatement(query, 'DIFFERENCE_TO')

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

    def _extractVariables(self, query, keyword='MODEL'):
        """ Extracts the names of the random variables for <keyword>-clause and returns it
        as a list of strings. The order is preserved.

        Note that it returns only strings, not actual Field objects.

        TODO: internally this function refers to self.models[query['FROM']].
            Remove this dependency
        """
        if keyword not in query:
            raise QuerySyntaxError("'{}'-statement missing".format(keyword))
        if query[keyword] == '*':
            return self.models[query['FROM']].names
        else:
            return query[keyword]

    def _extractPredict(self, query):
        """ Extracts from query the fields to predict and returns them in a
        list. The order is preserved.

        The Fields are dicts that are identical to the JSON in the query.
        """
        if 'PREDICT' not in query:
            raise QuerySyntaxError("'PREDICT'-statement missing")
        return query['PREDICT']

    def _extractDefaultValue(self, query):
        if 'DEFAULT_VALUE' not in query:
            return None
        else:
            return query['DEFAULT_VALUE']

    def _extractDefaultSubset(self, query):
        if 'DEFAULT_SUBSET' not in query:
            return None
        else:
            return query['DEFAULT_SUBSET']

    def _extractHide(self, query):
        if 'HIDE' not in query:
            return None
        else:
            return query['HIDE']

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

    def _extractDictByKeyword(self, query, keyword="OPTS"):
        if keyword not in query:
            return {}
        return query[keyword]

    def _extractOpts(self, query):
        return self._extractDictByKeyword(query)

    def _extractReload(self, query):
        if 'RELOAD' not in query:
            raise QuerySyntaxError("'RELOAD'-statement missing")
        what = query['RELOAD']
        if what == '*':
            return '*'
        else:
            raise NotImplementedError(
                'model-specific reloads not yet implemented!')

    def _extractSelect(self, query):
        if 'SELECT' not in query:
            raise QuerySyntaxError("'SELECT'-statement missing")
        if query['SELECT'] == '*':
            return self.models[query['FROM']].names
        else:
            return query['SELECT']

    def _automatically_create_model(self, model_name, query_opts):
        """Automatically create model or raise error if not possible.

        This option can be used in any query that references a model.
        Then the referenced model is created automatically based on the given parameters
        if it is missing in the modelbase.

        The syntax for this particular option in query opts is as follows:

        AUTO_CREATE_MODEL: {
            MODEL_TYPE: string, defaults to "empirical"
                The model type to be created. One of ["kde" "empirical"].
            FOR_MODEL: string
                The name/id of the model to create this new model for.
            }
        :return:
        """
        opts = query_opts.get('AUTO_CREATE_MODEL', None)
        if opts is None:
            raise QueryValueError(
                "The specified model does not exist: " + model_name)

        for_model_name = opts.get('FOR_MODEL', None)
        if for_model_name is None:
            raise QueryValueError(
                "Missing FOR_MODEL parameter in AUTO_CREATE_MODEL option statement")
        for_model = self.models.get(for_model_name, None)
        if for_model is None:
            raise QueryValueError('model {} does not exist and hence an empirical model cannot'
                                  ' be created for it.'.format(for_model_name))
        model_type = opts.get('MODEL_TYPE', 'empirical')

        data_model = model_fitting.make_empirical_model(modelname=model_name, base_model=for_model,
                                                     output_directory=self.model_dir, model_type=model_type)
        return self.add(data_model, name=model_name, filename=model_name+".mdl")

    def model_key(self, query) -> str:
        """A method that computes a key for the model needed by the query.

        Args:
            query: A PQL query.
        Returns:
            The key as a string.
        """

        base = query["AS"]
        model = self._extractVariables(query)
        where = self._extractWhere(query)

        key = str(base) + ':' + str(model).strip('[]') + ':' + str(where)
        key = key.strip('[]')
        key = key.replace(' ', '').replace('\'', '')

        return key

    def predict_key(self, query) -> str:
        """A method that computes a key for the prediction query needed by the query.

        Args:
            query: A PQL query.
        Returns:
            The key as a string.
        """

        base = self._extractFrom(query)
        predict = self._extractPredict(query)
        where = self._extractWhere(query)
        split = self._extractSplitBy(query)

        key = str(base) + str(predict) + str(where) + str(split)

        return key


if __name__ == '__main__':
    import models as md

    mb = ModelBase("mymb")
    iris = mb.models['iris']
    i2 = iris.copy()
    i2.marginalize(remove=["sepal_length"])
    print(i2.aggregate(method='maximum'))
    print(i2.aggregate(method='average'))
    aggr = md.AggregationTuple(
        ['sepal_width', 'petal_length'], 'maximum', 'petal_length', [])
    print(aggr)

    # foo = cc.copy().model(["total", "alcohol"], [md.ConditionTuple("alcohol", "equals", 10)])
    print(str(iris))
