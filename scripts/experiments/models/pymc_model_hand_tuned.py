#!usr/bin/python
# -*- coding: utf-8 -*-import string

import pymc3 as pm
import theano
import pandas as pd
import numpy as np

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper
from mb_modelbase.utils.Metrics import cll_allbus

import scripts.experiments.allbus as allbus_data

# LOAD FILES
test_data = allbus_data.train(numeric_happy=False)
train_data = allbus_data.test(numeric_happy=False)

df = train_data

# SAVE PARAMETER IN THIS FILE
model_file = 'allbus_results.dat'

sample_size = len(train_data)

allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                       'lived_abroad': {'No': 0, 'Yes': 1}}

allbus_backward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                      'lived_abroad': {0: 'No', 1: 'Yes'}}

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)

######################################
# hand tuned model
#####################################
def create_hand_tuned_model(modelname='allbus_hand_tuned', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data = train_data
    data = data.drop(['lived_abroad', 'health'], axis=1)

    # Reduce size of data to improve performance
    #data = data.sample(n=800, random_state=1)
    #data.sort_index(inplace=True)

    # Set up shared variables
    age = theano.shared(np.array(data['age']))
    age_min = np.min(data['age'])
    age_max = np.max(data['age'])
    age_diff = age_max-age_min

    educ_diff = 4
    inc_max = np.max(data['income'])

    sex_transformed = [allbus_forward_map['sex'][x] for x in data['sex']]
    eastwest_transformed = [allbus_forward_map['eastwest'][x] for x in data['eastwest']]

    allbus_model = pm.Model()
    with allbus_model:
        age_mu = pm.Uniform('age_mu', age_min, age_max)
        age_sigma = pm.Uniform('age_sigma', 1, 50)
        age = pm.TruncatedNormal('age', mu=age_mu, sigma=age_sigma, lower=age_min, upper=age_max, observed=data['age'])

        educ_p = pm.Dirichlet('educ_p', np.ones(5), shape=5)
        educ = pm.Categorical('educ', p=educ_p, observed=data["educ"], shape=1)

        sex_p = pm.Dirichlet('sex_p', np.ones(2), shape=2)
        sex = pm.Categorical('sex', p=sex_p, observed=sex_transformed, shape=1)

        eastwest_p = pm.Dirichlet('eastwest_p', np.ones(2), shape=2)
        eastwest = pm.Categorical('eastwest', p=eastwest_p, observed=eastwest_transformed, shape=1)

        # priors
        inc_mu_base = pm.Uniform('inc_mu_base', 1, 1000)
        inc_mu_age = pm.Uniform('inc_mu_age', 1, 1000)
        inc_mu_educ = pm.Uniform('inc_mu_educ', 1, 2000)
        inc_mu_sex = pm.Uniform('inc_mu_sex', 1, 1000)
        inc_mu_male_west = pm.Uniform('inc_mu_male_west', 1, 1000)
        inc_mu_eastwest = pm.Uniform('inc_mu_eastwest', 1, 1000)
        inc_mu = (
            inc_mu_base +
            inc_mu_age*(age-age_min)/age_diff +
            inc_mu_educ*educ/educ_diff +
            inc_mu_sex*sex +
            inc_mu_eastwest*eastwest +
            inc_mu_male_west*sex*eastwest
            )
        inc_sigma_base = pm.Uniform('inc_sigma_base', 100, 1000)
        inc_sigma_age = pm.Uniform('inc_sigma_age', 1, 1000)
        #inc_sigma_age_max = pm.Uniform('inc_sigma_age_max', 10,90)
        inc_sigma_educ = pm.Uniform('inc_sigma_educ', 1, 1000)
        inc_sigma_sex = pm.Uniform('inc_sigma_sex', 1, 2000)
        inc_sigma_eastwest = pm.Uniform('inc_sigma_eastwest', 1, 2000)
        inc_sigma = (
            inc_sigma_base +
            inc_sigma_age*(age-age_min)/age_diff +
            inc_sigma_educ*educ/educ_diff +
            inc_sigma_sex*sex +
            inc_sigma_eastwest*eastwest
            )

        # likelihood
        income = pm.Gamma('income', mu=inc_mu, sigma=inc_sigma, observed=data['income'])

        # priors happiness
        hap_mu_base = pm.Uniform('hap_mu_base', 0, 10)
        hap_mu_income = pm.Uniform('hap_mu_income', 0, 20)
        hap_mu_sex = pm.Uniform('hap_mu_sex', -10, 10)
        hap_mu_eastwest = pm.Uniform('hap_mu_eastwest', 0, 10)
        hap_mu = (
            hap_mu_base +
            hap_mu_income*(income/inc_max) +
            hap_mu_sex*sex +
            hap_mu_eastwest*eastwest
            )
        hap_sigma_base = pm.Uniform('hap_sigma_base', 0, 5)
        hap_sigma_income = pm.Uniform('hap_sigma_income', -3, 5)
        hap_sigma_sex = pm.Uniform('hap_sigma_sex', -1, 5)
        hap_sigma_eastwest = pm.Uniform('hap_sigma_eastwest', 0, 5)
        hap_sigma = (
            hap_sigma_base +
            hap_sigma_income*(income/inc_max) +
            hap_mu_sex*sex +
            hap_sigma_eastwest*eastwest
            )

        # likelihood happiness
        happiness = pm.TruncatedNormal('happiness', mu=hap_mu, sigma=hap_sigma, lower=0, upper=10, observed=data['happiness'])

    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={}, data_mapping=dtm, nr_of_posterior_samples=sample_size)

    if fit:
        m.fit(data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return data, m