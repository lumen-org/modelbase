# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Generic Test Suite for the model classes.

Generically tests model classes by running automatically generated queries on them. Essentially it only tests that
these queries go through without any exception being raised. It does not test if the returned results are correct as
such and it is not intended for that.

Such tests must be model specific and are hence found in the corresponding model specific test scripts.

What queries are generated?

 * 0th order aggregations and density queries (a&d queries): executed against the full model
 * 1st order a&d queries: executed against many(*) marginal models based on the full model
 * 1st order a&d queries: executed against many(**) conditional models based on the full model
 * 2nd order a&d queries: executed against many(*) marginal models based on the above 1st order models
 * 2nd order a&d queries: executed against many(*) conditional models based on the above 1st order models
 * 3rd order: continues like this

Thinking about useful console output for this test module:

  * output should help to identify where and when an uncaught exception was raised
  * most important information regarding this is:
    * what operation did you try to do?
    * what operation did you try tot do?
  * also useful as context information:
    * if operation was successful: what it its result?

"""
import logging
import random
import pandas as pd

from mb_modelbase.models_core import domains as dm
from mb_modelbase.models_core import models as md
from mb_modelbase.models_core.mockup_model import MockUpModel
from mb_modelbase.models_core.categoricals import CategoricalModel
from mb_modelbase.models_core.gaussians import MultiVariateGaussianModel as GaussianModel
from mb_modelbase.models_core.mixture_gaussians import MixtureOfGaussiansModel
from mb_modelbase.models_core.cond_gaussians import ConditionallyGaussianModel as CGModel
from mb_modelbase.models_core.cond_gaussian_wm import CgWmModel as CGWMModel
from mb_modelbase.models_core.mixable_cond_gaussian import MixableCondGaussianModel as MCGModel
from mb_modelbase.models_core.spflow import SPNModel
from mb_modelbase.models_core.empirical_model import EmpiricalModel
from mb_data.allbus import allbus

# from mb_modelbase.utils import is_running_in_debug_mode
# if is_running_in_debug_mode():
#     import mb_modelbase.models_core.models_debug

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)

def _values_of_extents(extents):
    """Returns a list of list, where each list consists of 'valid' values from the given extents.
    """
    def middle(domain):
        values = domain.values()
        if isinstance(domain, dm.NumericDomain):
            return (values[0]+values[1])/2
        elif isinstance(domain, dm.DiscreteDomain):
            n = len(values)
            return values[n // 2]
        else:
            raise TypeError("unknown type of domain : " + domain.__class__.__name__)
    return [[extent.values()[0] for extent in extents],
            [extent.values()[-1] for extent in extents],
            [middle(extent) for extent in extents]]


# stateful flag to set which aggregations and densities are to be tested
density_aggregation_flags = {'aggregations': True, 'density': True, 'density_sum': False}

def _test_all_density_and_aggregations(model, info):
    use = density_aggregation_flags
    if use['aggregations']:
        _test_aggregations(model, info=info)
    if use['density']:
        _test_density(model, info=info)
    if use['density_sum']:
        test_density_sum(model, info=info)

def _test_aggregations(model, info):
    """Computes all available aggregations on the given model.
    """
    logger.debug("(" + str(info) + ") Testing aggregations of " + model.name)
    for aggr_method in model._aggrMethods:
        a = model.aggregate(aggr_method)


def _test_density(model, info):
    """Computes densities on the current model.
    It runs three density queries:
     * at the 'lowest/first' value of each field's extent
     * at the 'highest/last' value of each field's extent
     * at the 'middle/average' value of each field's extent
     """
    logger.debug("(" + str(info) + ") Testing density of " + model.name)
    values = _values_of_extents(model.extents)
    for value in values:
        p = model.density(values=value)


def test_density_sum(model, info="", split_cnt=200, eps=0.05):
    # Calculates the accumulated 1d density along this 1-dimensional model and succeeds iff this adds up to 1 +- eps.
    if model.dim != 1:
        # raise NotImplementedError('Currently this can only test density for 1d models/data')
        return

    logger.debug("(" + str(info) + ") Testing density sum of " + model.name)

    f = model.fields[0]

    p = model.predict(predict=[md.Probability(f)],
                      splitby=[md.Split(f, args=split_cnt)])
    p_sum = p.sum().values
    if abs(p_sum - 1) > eps:
        raise AssertionError("Probability does not add up to 1. It is " + str(p_sum))

def _test_marginalization_mixed(model, depth, info):
    logger.debug("# (" + str(info) + ") Testing marginalization of " + model.name)
    depth -= 1
    info += "m"
    # categorical and continuous names
    cat = [f['name'] for f in model.fields if f['dtype'] == 'string']
    num = [f['name'] for f in model.fields if f['dtype'] == 'numerical']

    # run marginalize queries
    # loop over number of fields to remove at once: 1 to model.dim-1 many
    for n in range(1, model.dim):
        # loop over how many of the fields to remove are categorical
        for cat_n in range(0, n):
            num_n = n - cat_n

            # create copy
            m = model.copy()

            # names for the query
            names_to_marginalize = cat[:cat_n] + num[:num_n]
            names_to_keep = m.inverse_names(names_to_marginalize)

            # shuffle
            random.shuffle(names_to_keep)

            # derive marginal model
            m = m.model(names_to_keep)

            # try aggregations and density
            _test_all_density_and_aggregations(m, info)

            # recurse down?
            if depth > 0:
                _test_marginalization_mixed(m, 1, info)
                _test_conditioning_mixed(m, depth, info)


def _test_marginalization_discrete(model, depth, info):
    logger.info("# (" + str(info) + ") Testing marginalization of " + model.name)
    depth -= 1
    info += "m"
    # run marginalize queries
    # loop over number of fields to remove at once: 1 to model.dim-1 many
    for n in range(1, model.dim):
        # create copy
        m = model.copy()

        # names for the query
        names_to_marginalize = m.names[:n]
        names_to_keep = m.inverse_names(names_to_marginalize)

        # shuffle
        random.shuffle(names_to_keep)

        # derive marginal model
        m = m.model(names_to_keep)

        # try aggregations and density
        _test_all_density_and_aggregations(m, info)

        # recurse down?
        if depth > 0:
            _test_marginalization_discrete(m, 1, info)
            _test_conditioning_discrete(m, depth, info)

_test_marginalization_continuous = _test_marginalization_discrete  # it really is the same

_test_marginalization = {
    'discrete': _test_marginalization_discrete,
    'continuous': _test_marginalization_continuous,
    'mixed': _test_marginalization_mixed
}


def _test_conditioning_mixed(model, depth, info):
    logger.info("# (" + str(info) + ") Testing conditioning of " + model.name)
    depth -= 1
    info += "c"

    # categorical and continuous names
    cat = [f['name'] for f in model.fields if f['dtype'] == 'string']
    num = [f['name'] for f in model.fields if f['dtype'] == 'numerical']

    # run condition queries
    # loop over number of fields to condition out at once: 1 to model.dim-1 many
    for n in range(1, model.dim):
        # loop over how many of the fields to condition out are categorical
        for cat_n in range(0, n):
            num_n = n - cat_n

            # conditions
            names_to_condition_out = cat[:cat_n] + num[:num_n]
            extents_to_condition_out = [model.extents[idx] for idx in model.asindex(names_to_condition_out)]
            values_to_condition_on = _values_of_extents(extents_to_condition_out)

            # names to keep
            names_to_keep = model.inverse_names(names_to_condition_out)
            random.shuffle(names_to_keep)

            for val in values_to_condition_on:
                conditions = list(zip(names_to_condition_out, ["=="] * len(names_to_condition_out), val))

                # derive marginal model on copy
                m = model.copy().model(model=names_to_keep, where=conditions)

                # try aggregations and density
                _test_all_density_and_aggregations(m, info)

                # recurse down?
                if depth > 0:
                    _test_conditioning_mixed(m, 1, info)
                    _test_marginalization_mixed(m, depth, info)


def _test_conditioning_discrete(model, depth, info):
    logger.info("# (" + str(info) + ") Testing conditioning of " + model.name)
    depth -= 1
    info += "c"
    for n in range(1, model.dim):
        # conditions
        names_to_condition_out = model.names[:n]
        extents_to_condition_out = [model.extents[idx] for idx in model.asindex(names_to_condition_out)]
        values_to_condition_on = _values_of_extents(extents_to_condition_out)

        # names to keep
        names_to_keep = model.inverse_names(names_to_condition_out)
        random.shuffle(names_to_keep)

        for val in values_to_condition_on:
            conditions = list(zip(names_to_condition_out, ["=="] * len(names_to_condition_out), val))

            # derive marginal model on copy
            m = model.copy().model(model=names_to_keep, where=conditions)

            # try aggregations and density
            _test_all_density_and_aggregations(m, info)

            # recurse down?
            if depth > 0:
                _test_marginalization_discrete(m, depth, info)
                _test_conditioning_discrete(m, 1, info)

_test_conditioning_continuous = _test_conditioning_discrete  # it really is the same

_test_conditioning = {
    'discrete': _test_conditioning_discrete,
    'continuous': _test_conditioning_continuous,
    'mixed': _test_conditioning_mixed
}


def _test_all(models, models_setup, data, depth):
    for mode in ['discrete', 'continuous', 'mixed']:
        for model_class in models[mode]:
            # create and fit model
            model = model_class(name=model_class.__name__)

            # additional setup?
            if (mode, model_class) in models_setup:
                (models_setup[(mode, model_class)])(model)  # call it!

            # set data
            model.set_data(df=data[mode])

            # fit model
            model.fit()

            # test aggregations and such
            _test_all_density_and_aggregations(model, info="")

            # derive other models
            _test_marginalization[mode](model, depth, info="")
            _test_conditioning[mode](model, depth, info="")


if __name__ == '__main__':

    from mb_modelbase.utils import data_import_utils

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logging.root.setLevel(logger.level)

    # special model setup?
    models_setup = {
        ('continuous', MixtureOfGaussiansModel): lambda x: x.set_k(4)
    }

    # setup data for model training
    df = pd.read_csv('test_crabs.csv').drop(columns='index')
    all_, discrete, continuous = data_import_utils.get_columns_by_dtype(df)
    data_full = {
        'discrete': pd.DataFrame(df, columns=discrete),
        'continuous': pd.DataFrame(df, columns=continuous),
        'mixed': df
    }

    ## mockup model tests
    models = {
        'discrete': [MockUpModel],
        'continuous': [MockUpModel],
        'mixed': []
    }
    density_aggregation_flags = {'aggregations': False, 'density': True, 'density_sum': False}
    #_test_all(models, models_setup, data_full, depth=3)

    ## dedicated MCG model test
    models = {
        'discrete': [],
        'c   ontinuous': [],
        'mixed': [MCGModel]
    }
    density_aggregation_flags = {'aggregations': True, 'density': True, 'density_sum': False}
    #_test_all(models, models_setup, data_full, depth=3)

    ## dedicated density_sum test
    df = pd.read_csv('test_crabs.csv', usecols=['sex', 'RW'])
    all_, discrete, continuous = data_import_utils.get_columns_by_dtype(df)
    data = {
        'discrete': pd.DataFrame(df, columns=discrete),
        'continuous': pd.DataFrame(df, columns=continuous),
        'mixed': df
    }
    density_aggregation_flags = {'aggregations': False, 'density': False, 'density_sum': True}
    # _test_all(models, models_setup, data, depth=2)

    ## RUN TESTS FOR ALL MODELS:
    models = {
        'discrete': [],
        'continuous': [],
        #'mixed': [EmpiricalModel]
        'mixed': [SPNModel]
    }
    models_setup = {

    }
    # _models= {
    #     'discrete': [MockUpModel, CategoricalModel],
    #     'continuous': [MockUpModel, GaussianModel, MixtureOfGaussiansModel],
    #     'continuous': [MockUpModel, GaussianModel],
    #     'mixed': [CGModel, CGWMModel],
    #     'mixed': [CGModel, CGWMModel, MCGModel]
    # }

    models = {
        'discrete': [],
        'continuous': [],
        #'mixed': [EmpiricalModel]
        'mixed': [SPNModel]
    }
    models_setup = {
        ('continuous', MixtureOfGaussiansModel): lambda x: x.set_k(4)
    }

    density_aggregation_flags = {
        'aggregations': True,
        'density': True,
        'density_sum': False
    }

    df = allbus.mixed()
    all_, discrete, continuous = data_import_utils.get_columns_by_dtype(df)
    data = {
        'mixed': df
    }
    models_setup = {

    }

    models_setup = {
        ('mixed', SPNModel): lambda x: x.set_var_types(allbus.spn_metatypes['philipp'])
    }

    _test_all(models, models_setup, data, depth=3)














