#!usr/bin/python
# -*- coding: utf-8 -*-import string

import pymc3 as pm
import theano
import theano.tensor as tt
import pandas as pd
import numpy as np

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper
from mb_modelbase.utils.Metrics import cll_allbus

import scripts.experiments.allbus as allbus_data

# LOAD FILES
test_data = allbus_data.test(numeric_happy=False)
train_data = allbus_data.train(numeric_happy=False)

df = train_data

# SAVE PARAMETER IN THIS FILE
model_file = 'allbus_results.dat'
happy_query_file = 'allbus_happiness_values.dat'
income_query_file="allbus_income_values.dat"

sample_size = 50000

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
def create_allbus_model_NH0_hInv(filename='allbus_preprocessed_happiness_gamma', modelname='allbus_model_NH0_hInv', fit=True):
    #
    # a very simple handwritten model without dependencies
    #
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data_train = allbus_data.train(numeric_happy=False, inverse_happy=True)
    data_test = allbus_data.test(numeric_happy=False, inverse_happy=True)

    lived_abroad_transformed = [allbus_forward_map['lived_abroad'][x] for x in data_train['lived_abroad']]
    eastwest_transformed = [allbus_forward_map['eastwest'][x] for x in data_train['eastwest']]
    sex_transformed = np.array([allbus_forward_map['sex'][x] for x in data_train['sex']])

    age_min = np.min(data_train['age'])
    age_max = np.max(data_train['age'])
    age_diff = age_max-age_min

    allbus_model = pm.Model()
    with allbus_model:
        age_mu = pm.Uniform('age_mu', 45, 60)
        age_sigma = pm.Uniform('age_sigma', 15, 25)
        age = pm.TruncatedNormal('age', mu=age_mu, sigma=age_sigma, lower=15, upper=100, observed=data_train['age'])

        sex_p = pm.Dirichlet('sex_p', np.ones(2), shape=2)
        sex = pm.Categorical('sex', p=sex_p, observed=sex_transformed, shape=1)

        eastwest_p = pm.Dirichlet('eastwest_p', np.ones(2), shape=2)
        eastwest = pm.Categorical('eastwest', p=eastwest_p, observed=eastwest_transformed, shape=1)

        lived_abroad_p = pm.Dirichlet('lived_abroad_p', np.ones(2), shape=2)
        lived_abroad = pm.Categorical('lived_abroad', p=lived_abroad_p, observed=lived_abroad_transformed, shape=1)

        # education
        educ_p = pm.Dirichlet('educ_p', np.ones(6), shape=6)
        educ = pm.Categorical('educ', p=educ_p, observed=data_train['educ'], shape=1)

        # income
        inc_mu = pm.Uniform('inc_mu', 1500, 2000)
        inc_sigma = pm.Uniform('inc_sigma', 1000, 1500)
        income = pm.Gamma('income', mu=inc_mu, sigma=inc_sigma, observed=data_train['income'])

        hap_mu = pm.Uniform('hap_mu', 2, 5)
        hap_sigma = pm.Uniform('hap_sigma', 1, 3)
        happiness = pm.Gamma('happiness', mu=hap_mu, sigma=hap_sigma, observed=data_train['happiness'])

        # health
        health_mu = pm.Uniform('health_mu', 3, 4)
        health_sigma = pm.Uniform('health_sigma', 0.5, 1.5)
        health = pm.TruncatedNormal('health', mu=health_mu, sigma=health_sigma, lower=0, upper=6, observed=data_train['health'])

    shared_vars={'happiness' : happiness}
    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={}, data_mapping=dtm, nr_of_posterior_samples=sample_size)

    if fit:
        m.fit(data_train, auto_extend=False)
        cll_allbus(m, data_test, model_file, happy_query_file, income_query_file)
    return data_train, m

def create_allbus_tabubiccg_adjusted(filename="", modelname="allbus_tabubiccg_adjusted", fit=True):
    #
    # adjusted distributions of automatically learned model
    #
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803, 0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347, 0.653])
        lived_abroad = pm.Categorical('lived_abroad',
                                      p=tt.switch(tt.eq(eastwest, 0), [0.8847, 0.1153], [0.7859, 0.2141]))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061),
                         sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        happiness = pm.TruncatedNormal('happiness',
                                       mu=tt.switch(tt.eq(eastwest, 0), educ * 0.2477 + 6.5963, educ * 0.2059 + 7.295),
                                       sigma=tt.switch(tt.eq(eastwest, 0), 1.7444, 1.7047), lower=0, upper=11,
                                       transform=None)
        income = pm.Gamma('income', mu=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0),
                                                                               educ * 154.9222 + happiness * 75.795 + 144.8802,
                                                                               educ * 326.6378 + happiness * 116.0607 + -279.4209),
                                                 tt.switch(tt.eq(sex, 0),
                                                           educ * 220.2771 + happiness * -0.0931 + 594.0865,
                                                           educ * 384.5272 + happiness * 184.258 + -380.217)),
                          sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 636.2516, 956.2709),
                                          tt.switch(tt.eq(sex, 0), 845.3302, 1437.4018)), transform=None)
        age = pm.TruncatedNormal('age', mu=tt.switch(tt.eq(eastwest, 0), educ * -4.7345 + income * 0.0 + 70.8893,
                                                     educ * -5.3423 + income * 0.0025 + 65.1793),
                                 sigma=tt.switch(tt.eq(eastwest, 0), 16.4303, 16.2479), lower=10, upper=100,
                                 transform=None)
        health = pm.TruncatedNormal('health',
                                    mu=age * -0.0161 + educ * 0.0921 + income * 0.0001 + happiness * 0.214 + 2.3658,
                                    sigma=0.8404, lower=0, upper=6, transform=None)

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, happy_query_file, income_query_file)
    return train_data, m

def create_allbus_tabubiccg_adjusted_2(filename="", modelname="allbus_tabubiccg_adjusted_2", fit=True):
    #
    # adjusted distributions of automatically learned model. alternative version where happiness is inverse gamma
    #
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347,0.653])
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(eastwest, 0), [0.8847,0.1153], [0.7859,0.2141]))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        happiness_inv = pm.Gamma('happiness_inv', mu=tt.switch(tt.eq(eastwest, 0), 10-(educ*0.2477+6.5963), 10-(educ*0.2059+7.295)), sigma=tt.switch(tt.eq(eastwest, 0), 1.7444, 1.7047), transform=None)
        happiness = pm.Deterministic('happiness', 10-happiness_inv)
        income = pm.Gamma('income', mu=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*154.9222+happiness*75.795+144.8802, educ*326.6378+happiness*116.0607+-279.4209), tt.switch(tt.eq(sex, 0), educ*220.2771+happiness*-0.0931+594.0865, educ*384.5272+happiness*184.258+-380.217)), sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 636.2516, 956.2709), tt.switch(tt.eq(sex, 0), 845.3302, 1437.4018)), transform=None)
        age = pm.TruncatedNormal('age', mu=tt.switch(tt.eq(eastwest, 0), educ*-4.7345+income*0.0+70.8893, educ*-5.3423+income*0.0025+65.1793), sigma=tt.switch(tt.eq(eastwest, 0), 16.4303, 16.2479), lower=10, upper=100, transform=None)
        health = pm.TruncatedNormal('health', mu=age*-0.0161+educ*0.0921+income*0.0001+happiness*0.214+2.3658, sigma=0.8404, lower=0, upper=6, transform=None)

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, happy_query_file, income_query_file)
    return train_data, m


def create_allbus_model_NH0(filename='test_allbus.csv', modelname='allbus_model_NH0', fit=True):
    #
    # a very simple handwritten model without dependencies
    #
    if fit:
        modelname = modelname + '_fitted'
    # Load and prepare data
    data_train = train_data
    data_test = test_data

    lived_abroad_transformed = [allbus_forward_map['lived_abroad'][x] for x in data_train['lived_abroad']]
    eastwest_transformed = [allbus_forward_map['eastwest'][x] for x in data_train['eastwest']]
    sex_transformed = np.array([allbus_forward_map['sex'][x] for x in data_train['sex']])

    age_min = np.min(data_train['age'])
    age_max = np.max(data_train['age'])
    age_diff = age_max - age_min

    allbus_model = pm.Model()
    with allbus_model:
        age_mu = pm.Uniform('age_mu', 45, 60)
        age_sigma = pm.Uniform('age_sigma', 15, 25)
        age = pm.TruncatedNormal('age', mu=age_mu, sigma=age_sigma, lower=15, upper=100, observed=data_train['age'])

        sex_p = pm.Dirichlet('sex_p', np.ones(2), shape=2)
        sex = pm.Categorical('sex', p=sex_p, observed=sex_transformed, shape=1)

        eastwest_p = pm.Dirichlet('eastwest_p', np.ones(2), shape=2)
        eastwest = pm.Categorical('eastwest', p=eastwest_p, observed=eastwest_transformed, shape=1)

        lived_abroad_p = pm.Dirichlet('lived_abroad_p', np.ones(2), shape=2)
        lived_abroad = pm.Categorical('lived_abroad', p=lived_abroad_p, observed=lived_abroad_transformed, shape=1)

        # education
        educ_p = pm.Dirichlet('educ_p', np.ones(6), shape=6)
        educ = pm.Categorical('educ', p=educ_p, observed=data_train['educ'], shape=1)

        # income
        inc_mu = pm.Uniform('inc_mu', 1500, 2000)
        inc_sigma = pm.Uniform('inc_sigma', 1000, 1500)
        income = pm.Gamma('income', mu=inc_mu, sigma=inc_sigma, observed=data_train['income'])

        # happiness
        hap_mu = pm.Uniform('hap_mu', 7, 9)
        hap_sigma = pm.Uniform('hap_sigma', 1, 3)
        happiness = pm.TruncatedNormal('happiness', mu=hap_mu, sigma=hap_sigma, lower=0, upper=11,
                                       observed=data_train['happiness'])

        # health
        health_mu = pm.Uniform('health_mu', 3, 4)
        health_sigma = pm.Uniform('health_sigma', 0.5, 1.5)
        health = pm.TruncatedNormal('health', mu=health_mu, sigma=health_sigma, lower=0, upper=6,
                                    observed=data_train['health'])

    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={}, data_mapping=dtm,
                                nr_of_posterior_samples=sample_size)

    if fit:
        m.fit(data_train, auto_extend=False)
        cll_allbus(m, data_test, model_file, happy_query_file, income_query_file)
    return data_train, m


def create_allbus_model_NH1(filename='allbus_simplified.csv', modelname='allbus_model_NH1', fit=True):
    #
    # A complex handwritten model
    #
    if fit:
        modelname = modelname + '_fitted'
    # Load and prepare data
    data_train = train_data
    data_test = test_data

    lived_abroad_transformed = [allbus_forward_map['lived_abroad'][x] for x in data_train['lived_abroad']]
    eastwest_transformed = [allbus_forward_map['eastwest'][x] for x in data_train['eastwest']]
    sex_transformed = np.array([allbus_forward_map['sex'][x] for x in data_train['sex']])

    age_min = np.min(data_train['age'])
    age_max = np.max(data_train['age'])
    age_diff = age_max - age_min

    inc_max = np.max(data_train['income'])

    allbus_model = pm.Model()
    with allbus_model:
        lived_abroad_p = pm.Dirichlet('lived_abroad_p', np.ones(2), shape=2)
        lived_abroad = pm.Categorical('lived_abroad', p=lived_abroad_p, observed=lived_abroad_transformed, shape=1)

        eastwest_p = pm.Dirichlet('eastwest-p', np.ones(2), shape=2)
        eastwest = pm.Categorical('eastwest', p=eastwest_p, observed=eastwest_transformed, shape=1)

        educ_p = pm.Dirichlet('educ_p', np.ones(6), shape=6)
        educ = pm.Categorical('educ', p=educ_p, observed=data_train["educ"], shape=1)

        sex_p = pm.Dirichlet('sex-p', np.ones(2), shape=2)
        sex = pm.Categorical('sex', p=sex_p, observed=sex_transformed, shape=1)

        age_mu_base = pm.Uniform('age_mu_base', age_min, age_max)
        age_mu_sex = pm.Uniform('age_mu_sex', -10, 10)
        age_mu_educ = pm.Uniform('age_mu_educ', -10 * np.ones(6), 10 * np.ones(6), shape=6)
        age_mu = (age_mu_base +
                  age_mu_sex * sex +
                  age_mu_educ[educ]
                  )
        age_sigma_base = pm.Uniform('age_sigma_base', 1, 50)
        age_sigma_sex = pm.Uniform('age_sigma_sex', 1, 10)
        age_sigma_educ = pm.Uniform('age_sigma_educ', 1 * np.ones(6), 50 * np.ones(6), shape=6)
        age_sigma = (age_sigma_base +
                     age_sigma_sex * sex +
                     age_sigma_educ[educ]
                     )
        age = pm.TruncatedNormal('age', mu=age_mu, sigma=age_sigma, lower=age_min, upper=age_max,
                                 observed=data_train['age'])

        # priors income
        inc_mu_base = pm.Uniform('inc-mu-base', 1, 1000)
        inc_mu_age = pm.Uniform('inc-mu-age', 1, 1000)
        inc_mu_educ = pm.Uniform('inc-mu-educ', 1, 2000)
        inc_mu_sex = pm.Uniform('inc-mu-sex', 1, 1000)
        inc_mu_eastwest = pm.Uniform('inc-mu-eastwest', 1, 1000)
        inc_mu_male_west = pm.Uniform('inc-mu-male-west', 1, 1000)
        inc_mu = (
                inc_mu_base +
                inc_mu_age * (age_diff - abs(age - 45)) / age_diff +
                inc_mu_educ * educ / 5 +
                inc_mu_sex * sex +
                inc_mu_eastwest * eastwest +
                inc_mu_male_west * eastwest * sex
        )
        # pi_mu_gender = pm.Uniform('pi_mu_gender', -1*np.ones(2), 5*np.ones(2), shape=2)
        inc_sigma_base = pm.Uniform('inc-sigma-base', 1, 200)
        inc_sigma_age = pm.Uniform('inc-sigma-age', 50, 500)
        inc_sigma_educ = pm.Uniform('inc-sigma-educ', 50 * np.ones(6), 1200 * np.ones(6), shape=6)
        inc_sigma_sex = pm.Uniform('inc-sigma-sex', 1, 300)
        inc_sigma_eastwest = pm.Uniform('inc-sigma-eastwest', 1, 300)
        inc_sigma_male_west = pm.Uniform('inc-sigma-male-west', 100, 800)
        inc_sigma = (
                inc_sigma_base +
                inc_sigma_age * (age_diff - abs(age - 60)) / age_diff +
                inc_sigma_educ[educ] +
                inc_sigma_sex * sex +
                inc_sigma_eastwest * eastwest +
                inc_sigma_male_west * eastwest * sex
        )

        # likelihood income
        income = pm.Gamma('income', mu=inc_mu, sigma=inc_sigma, observed=data_train['income'])

        # priors happiness
        hap_mu_base = pm.Uniform('hap-mu-base', 0, 10)
        hap_mu_income = pm.Uniform('hap-mu-income', 0, 20)
        hap_mu_sex = pm.Uniform('hap-mu-sex', -10, 10)
        hap_mu_eastwest = pm.Uniform('hap-mu-eastwest', 0, 10)
        hap_mu = (
                hap_mu_base +
                hap_mu_income * (income / inc_max) +
                hap_mu_sex * sex +
                hap_mu_eastwest * eastwest
        )
        hap_sigma_base = pm.Uniform('hap-sigma-base', 0, 5)
        hap_sigma_income = pm.Uniform('hap-sigma-income', 1, 5)
        hap_sigma_sex = pm.Uniform('hap-sigma-sex', 1, 5)
        hap_sigma_eastwest = pm.Uniform('hap-sigma-eastwest', 0, 5)
        hap_sigma = (
                hap_sigma_base +
                hap_sigma_income * (income / inc_max) +
                hap_mu_sex * sex +
                hap_sigma_eastwest * eastwest
        )

        # likelihood happiness
        happiness = pm.TruncatedNormal('happiness', mu=hap_mu, sigma=hap_sigma, lower=0, upper=10,
                                       observed=data_train['happiness'])

        # health
        health_mu_base = pm.Uniform('health_mu_base', 2, 4)
        health_mu = (
            health_mu_base
        )
        health_sigma = pm.Uniform('health_sigma', 0, 2)
        health = pm.TruncatedNormal('health', mu=health_mu, sigma=health_sigma, lower=0, upper=6,
                                    observed=data_train['health'])

    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={}, data_mapping=dtm,
                                nr_of_posterior_samples=sample_size)

    if fit:
        m.fit(data_train, auto_extend=False)
        cll_allbus(m, data_test, model_file, happy_query_file, income_query_file)
    return data_train, m


def create_allbus_model_NH2(filename='test_allbus.csv', modelname='allbus_model_NH2', fit=True):
    #
    # An alternative complex handwritten model
    #
    if fit:
        modelname = modelname + '_fitted'
    # Load and prepare data
    data_train = train_data
    data_test = test_data

    lived_abroad_transformed = [allbus_forward_map['lived_abroad'][x] for x in data_train['lived_abroad']]
    eastwest_transformed = [allbus_forward_map['eastwest'][x] for x in data_train['eastwest']]
    sex_transformed = np.array([allbus_forward_map['sex'][x] for x in data_train['sex']])

    age_min = np.min(data_train['age'])
    age_max = np.max(data_train['age'])
    age_diff = age_max - age_min

    inc_max = np.max(data_train['income'])

    allbus_model = pm.Model()
    with allbus_model:
        sex_p = pm.Dirichlet('sex_p', np.ones(2), shape=2)
        sex = pm.Categorical('sex', p=sex_p, observed=sex_transformed, shape=1)

        eastwest_p = pm.Dirichlet('eastwest_p', np.ones(2), shape=2)
        eastwest = pm.Categorical('eastwest', p=eastwest_p, observed=eastwest_transformed, shape=1)

        lived_abroad_p = pm.Dirichlet('lived_abroad_p', np.ones(2), shape=2)
        lived_abroad = pm.Categorical('lived_abroad', p=lived_abroad_p, observed=lived_abroad_transformed, shape=1)

        # education
        educ_p = pm.Dirichlet('educ_p', np.ones(6), shape=6)
        educ = pm.Categorical('educ', p=educ_p, observed=data_train['educ'], shape=1)

        age_mu_base = pm.Uniform('age_mu_base', age_min, age_max)
        age_mu_sex = pm.Uniform('age_mu_sex', -10, 10)
        age_mu_educ = pm.Uniform('age_mu_educ', -10 * np.ones(6), 10 * np.ones(6), shape=6)
        age_mu = (age_mu_base +
                  age_mu_sex * sex +
                  age_mu_educ[educ]
                  )
        age_sigma_base = pm.Uniform('age_sigma_base', 1, 50)
        age_sigma_sex = pm.Uniform('age_sigma_sex', 1, 10)
        age_sigma_educ = pm.Uniform('age_sigma_educ', 1 * np.ones(6), 50 * np.ones(6), shape=6)
        age_sigma = (age_sigma_base +
                     age_sigma_sex * sex +
                     age_sigma_educ[educ]
                     )
        age = pm.TruncatedNormal('age', mu=age_mu, sigma=age_sigma, lower=age_min, upper=age_max,
                                 observed=data_train['age'])

        # income
        # depends on educ and on male+west
        inc_mu_base = pm.Uniform('inc_mu_base', 400, 1200)
        inc_mu_educ = pm.Uniform('inc_mu_educ', 1 * np.ones(6), 1500 * np.ones(6), shape=6)
        inc_mu_malewest = pm.Uniform('inc_mu_malewest', 1, 1000)
        inc_mu = inc_mu_base + inc_mu_educ[educ] + inc_mu_malewest * sex * eastwest
        inc_sigma_base = pm.Uniform('inc_sigma_base', 200, 500)
        inc_sigma_educ = pm.Uniform('inc_sigma_educ', 1 * np.ones(6), 1500 * np.ones(6), shape=6)
        inc_sigma_malewest = pm.Uniform('inc_sigma_malewest', 1, 1000)
        inc_sigma = inc_sigma_base + inc_sigma_educ[educ] + inc_sigma_malewest * sex * eastwest
        income = pm.Gamma('income', mu=inc_mu, sigma=inc_sigma, observed=data_train['income'])

        # health
        # depends on age and educ
        health_mu_base = pm.Uniform('health_mu_base', 0, 5)
        health_mu_age = pm.Uniform('health_mu_age', -2, 0)
        health_mu_educ = pm.Uniform('health_mu_educ', 0, 2)
        health_mu = health_mu_base + health_mu_age * (age - age_min) / age_diff + health_mu_educ * educ / 5
        health_sigma_base = pm.Uniform('health_sigma_base', 0.5, 1.5)
        health_sigma_inc = pm.Uniform('health_sigma_inc', -0.5, 0)
        health_sigma = health_sigma_base + health_sigma_inc * income / inc_max
        health = pm.TruncatedNormal('health', mu=health_mu, sigma=health_sigma, lower=0.9, upper=5.1,
                                    observed=data_train['health'])

        # happiness
        # depends on income and health,sex,eastwest,educ
        hap_mu_base = pm.Uniform('hap_mu_base', 0, 10)
        hap_mu_income = pm.Uniform('hap_mu_income', 0, 5)
        hap_mu_health = pm.Uniform('hap_mu_health', 0, 5)
        hap_mu_sex = pm.Uniform('hap_mu_sex', 0 * np.ones(2), 3 * np.ones(2), shape=2)
        hap_mu_eastwest = pm.Uniform('hap_mu_eastwest', 0 * np.ones(2), 3 * np.ones(2), shape=2)
        hap_mu_educ = pm.Uniform('hap_mu_educ', 0, 3)
        hap_mu = hap_mu_base + hap_mu_income * income / inc_max + hap_mu_health * health / 5 + hap_mu_sex[sex] + \
                 hap_mu_eastwest[eastwest] + hap_mu_educ * educ / 5
        hap_sigma_base = pm.Uniform('hap_sigma_base', 1, 5)
        hap_sigma_income = pm.Uniform('hap_sigma_income', -0.5, 0.5)
        hap_sigma = hap_sigma_base + hap_sigma_income * income / inc_max
        happiness = pm.TruncatedNormal('happiness', mu=hap_mu, sigma=hap_sigma, lower=0, upper=10.1,
                                       observed=data_train['happiness'])

    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={}, data_mapping=dtm,
                                nr_of_posterior_samples=sample_size)

    if fit:
        m.fit(data_train, auto_extend=False)
        cll_allbus(m, data_test, model_file, happy_query_file, income_query_file)
    return data_train, m
