# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Generic Test Suite for the model classes.

Generically tests model classes by running automatically generated queries on them. Essentially it only tests that
these queries go through without any exception being raised. It does not test if the returned results are correct as
such and it is not intended for that.

Such tests must be model specific and are hence found in the corresponding model specific test scripts.


General idea for this test suite:

base of all testing
 * a 6D data set of 3 categorical and 3 continuous random variables

maintain list of model classes seperated by:
 * continuous, discrete or mixed model?

how to test each model class:
 1. train model on data
 2. run tests

about these tests:
 * they are generic, i.e. queries are generated at runtime depending on available fields in model

"""

import domains as dm

import data.crabs.crabs as crabs

# model classes
models = {
    'discrete' : [],
    'continuous': [],
    'mixed': []
}


def test_aggregations(model):
    """Computes all available aggregations on the given model."""
    for aggr_method in model._aggrMethods:
        model.aggregate(aggr_method)


def test_density(model):
    """Computes densities on the current model.
    It runs three density queries:
     * at the 'lowest/first' value of each field's extent
     * at the 'highest/last' value of each field's extent
     * at the 'middle/average' value of each field's extent
     """

    extents = model.extents

    def middle(domain):
        values = domain.values()
        if isinstance(dm.NumericDomain):
            return (values[0]+values[1])/2
        elif isinstance(dm.DiscreteDomain):
            n = len(values)
            return values[n // 2]
        else:
            raise TypeError("unknown type of domain : " + domain.__class__.__name__)

    lowest = [extent.values()[0] for extent in extents]
    model.density(lowest)

    highest = [extent.values()[-1] for extent in extents]
    model.density(highest)

    middle = [middle(extent) for extent in extents]
    model.density(middle)


def test_marginalization_mixed(model):
    # find out categorical and continuous dimensions and create marginalize queries at runtime:
    #


    # derive marginal submodel: remove 1 categorical, 1 continuous
    # run aggregations:

    # derive marginal submodel: remove 2 categorical, 1 continuous
    # derive marginal submodel: remove 1 categorical, 2 continuous
    # derive marginal submodel: remove 2 categorical, 2 continuous
    # derive marginal submodel: remove 3 categorical, 1 continuous
    # derive marginal submodel: remove 3 categorical, 2 continuous
    # derive marginal submodel: remove 1 categorical, 3 continuous
    # derive marginal submodel: remove 2 categorical, 3 continuous
    pass

def test_categorical():
    pass

def test_continuous():
    pass

def test_all():
    
    # setup data for model training
    df = crabs.mixed('data/crabs/australian-crabs.csv')

    #

    data = {
        'discrete': ,
        'continuous': ,
        'mixed': df
    }
    data_discrete = []
    data_continuous = []
    #
    # test_density = {
    #     'discrete': test_density_discrete,
    #     'continuous': test_density_continuous,
    #     'mixed': test_density_mixed
    # }

    for mode in ['discrete', 'continuous', 'mixed']:
        for model_class in models[mode]:
            # create and fit model
            model = model_class(name=model_class.__name__)
            model.fit(df=data)

            # test model
            test_aggregations(model)
            test_density(model)
            #test_marginalization(model)
            #test_conditioning(model)
    


"""
Generic Testing of models:

 what can I test generically?
  * it doesn't raise anything when it should not: run several queries against the model
  * it cannot however know any specific prediction answer of the model
  * I could even make up a testing suite that detects/knows:
    * mixed or not mixed model: run appropiate subqueries

need data of ~6D? 3 categorical, 3 continuous

def _




"""



