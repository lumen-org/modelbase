# Copyright (c) 2017-2018 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas
"""

import logging
import os.path
import pandas as pd

from mb_modelbase.models_core.empirical_model import EmpiricalModel
from mb_modelbase.server import ModelBase

# setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s :: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# resolve relative path of data for this script
path_prefix = os.path.join(os.path.dirname(__file__), os.pardir, 'mb_data')


def make_empirical_model(modelname, output_directory, input_file=None, df=None):
    """A one-stop function to create an `EmpiricalModel` from data.

    Args:
        modelname: str
            Name of the model. Will be used as the the name of the model and for the filename of the saved model.
        output_directory: str, optional.
            path of directory where to store the model (not file name!). If set to None, model will not be saved on
            filesystem.
        input_file: str, optional.
            path of csv file to read for data to use for training of model. Alternatively, directly specify the data
            in `df`.
        df: pd.DataFrame, optional.
            Data to use for model fitting. Alternatively specify a csv file in `input_file`.
    Return:
        The learned model.

    """
    if input_file is None and df is None:
        raise ValueError("you must specify at least one of `input_file` or `df`.")

    if input_file is not None:
        df = pd.read_csv(input_file, index_col=False, skip_blank_lines=True)
        print("read data from file {}".format(input_file))

    print("Your data frame looks like this: \n{}".format(str(df.head())))

    # preprocess
    df2 = df.dropna(axis=0, how="any")
    dropped_rows = df.shape[0] - df2.shape[0]
    if dropped_rows > 0:
        print("dropped {} rows due to nans in data".format(dropped_rows))

    # fit model
    model = EmpiricalModel(modelname)
    model.fit(df2)
    print("Successfully fitted empirical model!")

    # save
    if output_directory is not None:
        output_file = os.path.join(output_directory, modelname + ".mdl")
        output_file = os.path.abspath(output_file)
        model.save(output_file)
        print("Saved model in file: \n{}".format(output_file))

    return model


def save_models(models, directory):
    """Saves all models in dict `models` to given directory.

    Args:
        models (dict): Dictionary as returned by fit_models().
        directory (string): Optional, defaults to '../../models'. Path where to store all models.
    Returns:
        None
    """
    modelbase = ModelBase("refitter", load_all=False, model_dir=directory)
    for model in models.values():
        if model['status'] == 'SUCCESS':
            modelbase.add(model['model'])
    modelbase.save_all_models()


def fit_models(spec, verbose=False, include=None, exclude=None):
    """Fits models according to provided specs and returns a dict of the learned models.

    Args:
        spec (dict): Dictionary of <name> to model specifications. A single model specification may either be a dict or
            a callable (no arguments) that returns a dict. Either way, the configuration dict is as follows:
                * 'class': Usually <class-object of model> but can be any function that returns a model when called.
                * 'data': Optional. The data frame of data to use for fitting. If not spefified the 'class' is expected
                    to return a fitted model.
                * 'classopts': Optional. A dict passed as keyword-arguments to 'class'.
                * 'fitopts': Optional. A dict passed as keyword-arguments to the .fit method of the created model
                    instance.
            The idea of the callable is that delay data acquisition until model selection.
        verbose (bool): Optional. Defaults to False. More verbose logging iff set to true.
        include (list-like of strings): Optional. Defaults to None. List of models to explicitly include. By default
            all are included.
        exclude (list-like of strings): Optional. Defaults to None. List of models to explicitly exclude. By default
            none are excluded.

    Returns: A dict of <name> to dict of 'model' that contains the learned model, status that contains the status
        ('SUCCESS' or 'FAIL') and message that contains an optional message explaining the status.
    """

    if include is None:
        include = spec.keys()
    if exclude is None:
        exclude = set()
    include = set(include)
    found = set()   # set of names of models that we've found
    fitted = set()  # set of names of models that we've fitted
    failed = set()  # set of names of models that failed to fit for any reason

    models = {}
    for (id_, value) in spec.items():
        if id_ in include and id_ not in exclude:
            found.add(id_)
            try:
                config = {'classopts': {}, 'fitopts': {}, 'data': None}
                if callable(value):
                    # value() may be a function returning a suitable dict
                    value = value()
                config.update(value)
                Modelclass = config['class']
                df = config['data']
                if verbose:
                    logger.info("Fitting model for data set '" + str(id_) + "' ...")
                model = Modelclass(id_, **config['classopts'])
                if df is not None:
                    # only fit if data available. otherwise we expect that the model has been fitted elsewhere
                    model.fit(df, **config['fitopts'])
                models[id_] = {
                    'model': model,
                    'status': 'SUCCESS'
                }
                if verbose:
                    logger.info("... done.")
                fitted.add(id_)
            except:
                import traceback
                failed.add(id_)
                models[id_] = {
                    'model': None,
                    'status': 'FAIL',
                    'message': "Unexpected error: \n{}".format(id_, traceback.format_exc())
                }
                logger.warning("Failed to learn model '{}'! Unexpected error: \n{}".format(id_, traceback.format_exc()))
                logger.warning("I continue with the rest of the models anyway.")

    # check if all the models in include were found and fitted
    logger.info("Fitted " + str(len(fitted)) + " models in total: " + ("<none>" if len(fitted) == 0 else str(fitted)))

    if include != found:
        logger.warning("Some model you '--include'd were not found in the provided specification'.")
        logger.warning("The missing models are: " + str(include - found))
    if len(failed) != 0:
        logger.warning("Some models FAILED to learn! See above for details.")
        logger.warning("The failed models are: " + str(failed))
    if len(fitted) == 0:
        logger.error("I did not fit a single model! Something must be wrong!")
    return models
