# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Generic Test Suite for the model classes.

Generically tests model classes by running automatically generated queries on them. Essentially it only tests that
these queries go through without any exception being raised. It does not test if the returned results are correct as
such and it is not intended for that.

Such tests must be model specific and are hence found in the corresponding model specific test scripts.

"""
import unittest

import random

import pandas as pd

import domains as dm
import models as md
from mockup_model import MockUpModel
from categoricals import CategoricalModel
from gaussians import MultiVariateGaussianModel as GaussianModel
from mixture_gaussians import MixtureOfGaussiansModel
from cond_gaussians import ConditionallyGaussianModel as CGModel
from cond_gaussian_wm import CgWmModel as CGWMModel

import data.crabs.crabs as crabs

# REGISTER ALL YOUR MODEL SUBCLASSES TO TEST HERE
# model classes
# models = {
#     'discrete': [],
#     'continuous': [MixtureOfGaussiansModel],
#     'mixed': []
# }
models = {
    'discrete': [MockUpModel, CategoricalModel],
    'continuous': [MockUpModel, GaussianModel, MixtureOfGaussiansModel],
    'mixed': [MockUpModel, CGModel, CGWMModel]
}

model_setup = {
    ('continuous', MixtureOfGaussiansModel): lambda x: x.set_k(4)
}


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


def _test_aggregations(model):
    """Computes all available aggregations on the given model."""
    for aggr_method in model._aggrMethods:
        model.aggregate(aggr_method)


def _test_density(model):
    """Computes densities on the current model.
    It runs three density queries:
     * at the 'lowest/first' value of each field's extent
     * at the 'highest/last' value of each field's extent
     * at the 'middle/average' value of each field's extent
     """
    values = _values_of_extents(model.extents)
    for value in values:
        model.density(value)


def _test_marginalization_mixed(model):
    # categorical and continuous names
    cat = model._categoricals
    num = model._numericals

    # run marginalize queries
    # loop over number of fields to remove at once: 1 to model.dim-1 many
    for n in range(1, model.dim - 1):
        # loop over how many of the fields to remove are categorical
        for cat_n in range(1, n-1):
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
            _test_aggregations(m)
            _test_density(m)
    pass


def _test_marginalization_discrete(model):
    # run marginalize queries
    # loop over number of fields to remove at once: 1 to model.dim-1 many
    for n in range(1, model.dim - 1):
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
        _test_aggregations(m)
        _test_density(m)

_test_marginalization_continuous = _test_marginalization_discrete  # it really is the same

_test_marginalization = {
    'discrete': _test_marginalization_discrete,
    'continuous': _test_marginalization_continuous,
    'mixed': _test_marginalization_mixed
}


def _test_conditioning_mixed(model):
    # categorical and continuous names
    cat = model._categoricals
    num = model._numericals

    # run condition queries
    # loop over number of fields to condition out at once: 1 to model.dim-1 many
    for n in range(1, model.dim - 1):
        # loop over how many of the fields to condition out are categorical
        for cat_n in range(1, n - 1):
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
                _test_aggregations(m)
                _test_density(m)


def _test_conditioning_discrete(model):
    for n in range(1, model.dim - 1):
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
            _test_aggregations(m)
            _test_density(m)
            
_test_conditioning_continuous = _test_conditioning_discrete  # it really is the same

_test_conditioning = {
    'discrete': _test_conditioning_discrete,
    'continuous': _test_conditioning_continuous,
    'mixed': _test_conditioning_mixed
}


def test_all():
    # setup data for model training
    df = crabs.mixed('data/crabs/australian-crabs.csv')
    all_, discrete, continuous = md.get_columns_by_dtype(df)
    data = {
        'discrete': pd.DataFrame(df, columns=discrete),
        'continuous': pd.DataFrame(df, columns=continuous),
        'mixed': df
    }

    for mode in ['discrete', 'continuous', 'mixed']:
        for model_class in models[mode]:
            # create and fit model
            model = model_class(name=model_class.__name__)

            # additional setup?
            if (mode, model_class) in model_setup:
                (model_setup[(mode, model_class)])(model)  # call it!

            model.fit(df=data[mode])

            # test model
            _test_aggregations(model)
            _test_density(model)
            _test_marginalization[mode](model)
            _test_conditioning[mode](model)


class TestGeneric(unittest.TestCase):
    def test_first(self):
        test_all()

if __name__ == '__main__':
    unittest.main()
