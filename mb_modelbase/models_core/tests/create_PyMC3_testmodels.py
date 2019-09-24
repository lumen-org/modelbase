import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.models_core.empirical_model import EmpiricalModel
import theano
from scripts.run_conf import cfg as user_cfg
import os
import timeit
import scipy.stats
import math



######################################
# pymc3_testcase_model
#####################################
def create_pymc3_simplest_model(modelname='pymc3_simplest_model', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    np.random.seed(2)
    size = 100
    mu = np.random.normal(0, 1, size=size)
    sigma = 1
    X = np.random.normal(mu, sigma, size=size)
    data = pd.DataFrame({'X': X})

    basic_model = pm.Model()
    with basic_model:
        sigma = 1
        mu = pm.Normal('mu', mu=0, sd=sigma)
        X = pm.Normal('X', mu=mu, sd=sigma, observed=data['X'])
    m = ProbabilisticPymc3Model(modelname, basic_model)
    if fit:
        m.fit(data)
    return data, m

######################################
# pymc3_getting_started_model
######################################
def create_pymc3_getting_started_model(modelname='pymc3_getting_started_model', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    np.random.seed(123)
    alpha, sigma = 1, 1
    beta_0 = 1
    beta_1 = 2.5
    size = 100
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = alpha + beta_0 * X1 + beta_1 * X2 + np.random.randn(size) * sigma
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta_0 = pm.Normal('beta_0', mu=0, sd=10)
        beta_1 = pm.Normal('beta_1', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=5)

        # Expected value of outcome
        mu = alpha + beta_0 * data['X1'] + beta_1 * data['X2']

        # Likelihood (sampling distribution) of observations
        Y = pm.Normal('Y', mu=mu, sd=sigma, observed=data['Y'])
        X1 = pm.Normal('X1', mu=data['X1'], sd=sigma, observed=data['X1'])
        X2 = pm.Normal('X2', mu=data['X2'], sd=sigma, observed=data['X2'])

        m = ProbabilisticPymc3Model(modelname, basic_model)
        if fit:
            m.fit(data)
        return data, m

##############################################
# pymc3_getting_started_model_independent vars
##############################################
def create_pymc3_getting_started_model_independent_vars (
        modelname='pymc3_getting_started_model_independent_vars', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    np.random.seed(123)
    alpha, sigma = 1, 1
    beta_0 = 1
    beta_1 = 2.5
    size = 100
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = alpha + beta_0 * X1 + beta_1 * X2 + np.random.randn(size) * sigma
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
    X1 = theano.shared(X1)
    X2 = theano.shared(X2)

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta_0 = pm.Normal('beta_0', mu=0, sd=10)
        beta_1 = pm.Normal('beta_1', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value of outcome
        mu = alpha + beta_0 * X1 + beta_1 * X2

        # Likelihood (sampling distribution) of observations
        Y = pm.Normal('Y', mu=mu, sd=sigma, observed=data['Y'])

    m = ProbabilisticPymc3Model(modelname, basic_model, shared_vars={'X1': X1, 'X2': X2})
    if fit:
        m.fit(data)
    return data, m

###########################################################
# pymc3_getting_started_model_independent vars_nosharedvars
###########################################################
def create_pymc3_getting_started_model_independent_vars_nosharedvars (
        modelname='pymc3_getting_started_model_independent_vars_nosharedvars', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    np.random.seed(123)
    alpha, sigma = 1, 1
    beta_0 = 1
    beta_1 = 2.5
    size = 100
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = alpha + beta_0 * X1 + beta_1 * X2 + np.random.randn(size) * sigma
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
    X1 = theano.shared(X1)

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta_0 = pm.Normal('beta_0', mu=0, sd=10)
        beta_1 = pm.Normal('beta_1', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value of outcome
        mu = alpha + beta_0 * X1 + beta_1 * data['X2']

        # Likelihood (sampling distribution) of observations
        Y = pm.Normal('Y', mu=mu, sd=sigma, observed=data['Y'])
        m = ProbabilisticPymc3Model(modelname, basic_model, shared_vars={'X1': X1})
        if fit:
            m.fit(data)
        return data, m

######################################
# pymc3_coal_mining_disaster_model
######################################
def create_pymc3_coal_mining_disaster_model(modelname='pymc3_coal_mining_disaster_model', fit=True):
    if fit:
        modelname = modelname+'_fitted'

    disasters = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                2, 2, 3, 4, 2, 1, 3, 3, 2, 1, 1, 1, 1, 3, 0, 0,
                                1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    years = np.arange(1851, 1962)

    data = pd.DataFrame({'years': years, 'disasters': disasters})
    years = theano.shared(years)
    with pm.Model() as disaster_model:

        switchpoint = pm.DiscreteUniform('switchpoint', lower=years.min(), upper=years.max(),
        testval=1900)

        # Priors for pre- and post-switch rates number of disasters
        early_rate = pm.Exponential('early_rate', 1.0)
        late_rate = pm.Exponential('late_rate', 1.0)

        # Allocate appropriate Poisson rates to years before and after current
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

        disasters = pm.Poisson('disasters', rate, observed=data['disasters'])
        #years = pm.Normal('years', mu=data['years'], sd=0.1, observed=data['years'])

    m = ProbabilisticPymc3Model(modelname, disaster_model, shared_vars={'years': years})
    if fit:
        m.fit(data)
    return data, m

########################################
# eight_schools_model
########################################
def create_pymc3_eight_schools_model(modelname='pymc3_eight_schools_model', fit=True):
    if fit:
        modelname = modelname+'_fitted'

    scores = np.array([28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16])
    standard_errors = np.array([14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6])
    data = pd.DataFrame({'test_scores': scores, 'standard_errors': standard_errors})
    standard_errors = theano.shared(standard_errors)

    with pm.Model() as normal_normal_model:
        tau = pm.Uniform('tau',lower=0,upper=10)
        mu = pm.Uniform('mu',lower=0,upper=10)
        theta_1 = pm.Normal('theta_1', mu=mu, sd=tau)
        theta_2 = pm.Normal('theta_2', mu=mu, sd=tau)
        theta_3 = pm.Normal('theta_3', mu=mu, sd=tau)
        theta_4 = pm.Normal('theta_4', mu=mu, sd=tau)
        theta_5 = pm.Normal('theta_5', mu=mu, sd=tau)
        theta_6 = pm.Normal('theta_6', mu=mu, sd=tau)
        theta_7 = pm.Normal('theta_7', mu=mu, sd=tau)
        theta_8 = pm.Normal('theta_8', mu=mu, sd=tau)

        test_scores = pm.Normal('test_scores',
                                mu=[theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8],
                                sd=standard_errors, observed=data['test_scores'])

    m = ProbabilisticPymc3Model(modelname, normal_normal_model,
                                shared_vars={'standard_errors': standard_errors}, fixed_data_length=True)
    if fit:
        m.fit(data)
    return data, m

######################################
# getting_started_model_shape
######################################
def create_getting_started_model_shape(modelname='pymc3_getting_started_model_shape', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    np.random.seed(123)
    alpha, sigma = 1, 1
    beta_0 = 1
    beta_1 = 2.5
    size = 100
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = alpha + beta_0 * X1 + beta_1 * X2 + np.random.randn(size) * sigma
    data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
    X1 = theano.shared(X1)
    X2 = theano.shared(X2)

    basic_model = pm.Model()

    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta_0', mu=0, sd=10, shape=2)
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observations
        Y = pm.Normal('Y', mu=mu, sd=sigma, observed=data['Y'])

    m = ProbabilisticPymc3Model(modelname, basic_model, shared_vars={'X1': X1, 'X2': X2})
    if fit:
        m.fit(data)
    return data, m

######################################
# Flight delay models
######################################
def create_flight_delay_model_1(filename='airlineDelayDataProcessed.csv', modelname='flight_delay_1', fit=True):
    if fit:
        modelname = modelname+'_fitted'

    data = pd.read_csv(filename)
    data = data.rename(columns={'DEP_TIME': 'dep_time', 'DEP_DELAY': 'depdelay'})

    # Drop variables that are not considered in the model
    data = data.drop(['UNIQUE_CARRIER', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID',
                      'ACTUAL_ELAPSED_TIME', 'ARR_DELAY', 'DISTANCE'], axis=1)

    # Reduce size of data to improve performance
    data = data.sample(n=1000, random_state=1)
    data.sort_index(inplace=True)

    # Create shared variables
    deptime = theano.shared(np.array(data['dep_time']))

    # Create model
    delay_model = pm.Model()

    with delay_model:
        beta_dep = pm.Uniform('beta_dep', 0, 1, shape=2)
        var = pm.Uniform('var', 0, 100)
        # I assume that the depdelay is a function of deptime
        mu_depdelay = beta_dep[0] + beta_dep[1] * deptime
        depdelay = pm.Normal('depdelay', mu_depdelay, var, observed=data['depdelay'])

    m = ProbabilisticPymc3Model(modelname, delay_model, shared_vars={'dep_time': deptime})
    if fit:
        m.fit(data)
    return data, m

def create_flight_delay_model_2(filename='airlineDelayDataProcessed.csv', modelname='flight_delay_2', fit=True):
    if fit:
        modelname = modelname+'_fitted'

    data = pd.read_csv(filename)
    data = data.rename(columns={'DEP_TIME': 'dep_time', 'DEP_DELAY': 'depdelay'})

    # Drop variables that are not considered in the model
    data = data.drop(['UNIQUE_CARRIER', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID',
                      'ACTUAL_ELAPSED_TIME', 'ARR_DELAY', 'DISTANCE'], axis=1)

    # Reduce size of data to improve performance
    data = data.sample(n=1000, random_state=1)
    data.sort_index(inplace=True)

    # Create shared variables
    deptime = theano.shared(np.array(data['dep_time']))

    # Create model
    delay_model = pm.Model()

    with delay_model:
        beta_dep = pm.Uniform('beta_dep', 0, 1, shape=2)
        beta_var = pm.Uniform('beta_var', 0, 1, shape=2)
        # Improvement 1: Assume that variance is a linear function of time, instead of uniformly distributed
        var = beta_var[0] + beta_var[1] * deptime
        # I assume that depdelay is a function of deptime
        mu_depdelay = beta_dep[0] + beta_dep[1] * deptime
        depdelay = pm.Normal('depdelay', mu_depdelay, var, observed=data['depdelay'])

    m = ProbabilisticPymc3Model(modelname, delay_model, shared_vars={'dep_time': deptime})
    if fit:
        m.fit(data)
    return data, m

def create_flight_delay_model_3(filename='airlineDelayDataProcessed.csv', modelname='flight_delay_3', fit=True):
    if fit:
        modelname = modelname+'_fitted'

    data = pd.read_csv(filename)
    data = data.rename(columns={'DEP_TIME': 'dep_time', 'DEP_DELAY': 'depdelay'})

    # Drop variables that are not considered in the model
    data = data.drop(['UNIQUE_CARRIER', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID',
                      'ACTUAL_ELAPSED_TIME', 'ARR_DELAY', 'DISTANCE'], axis=1)

    # Reduce size of data to improve performance
    data = data.sample(n=1000, random_state=1)
    data.sort_index(inplace=True)

    # Create shared variables
    deptime = theano.shared(np.array(data['dep_time']))

    # Create model
    delay_model = pm.Model()

    with delay_model:
        beta_var = pm.Uniform('beta_var', 0, 1, shape=2)
        # Improvement 1: Assume that variance is a linear function of time, instead of uniformly distributed
        var = beta_var[0] + beta_var[1] * deptime
        # Improvement 3: Apply a shift to the data so that the HalfNormalDistribution fits better
        shift = min(data['depdelay'])
        # Improvement 2: I assume that depdelay is  bounded at 0 and only the variance is a function of deptime
        depdelay = pm.HalfNormal('depdelay', sd=var, observed=data['depdelay']-shift)

    m = ProbabilisticPymc3Model(modelname, delay_model, shared_vars={'dep_time': deptime})
    if fit:
        m.fit(data)
    return data, m



######################################
# allbus models
######################################
def create_allbus_model_1(filename='test_allbus.csv', modelname='allbus_model_1', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data = pd.read_csv(filename, index_col=0)
    data = data.drop(['eastwest', 'lived_abroad', 'spectrum', 'sex', 'educ', 'health'], axis=1)
    # Reduce size of data to improve performance
    data = data.sample(n=500, random_state=1)
    data.sort_index(inplace=True)
    # Set up shared variables
    age = theano.shared(np.array(data['age']))
    allbus_model = pm.Model()
    with allbus_model:
        # priors
        sd_income = pm.Uniform('sd_income', 0, 1000)
        alpha_inc = pm.Uniform('alpha_inc', -1000, 1000)
        beta_inc = pm.Uniform('beta_inc', -100, 100)
        sd_happ = pm.Uniform('sd_happ', 0, 5)
        alpha_happ = pm.Uniform('alpha_happ', -10, 10)
        beta_happ = pm.Uniform('beta_happ', -0.001, 0.001)
        # likelihood
        mu_income = alpha_inc + beta_inc * age
        income = pm.Normal('income', mu_income, sd_income, observed=data['income'])
        mu_happ = alpha_happ + beta_happ * income
        happiness = pm.Normal('happiness', mu_happ, sd_happ, observed=data['happiness'])

    # Create model instance for Lumen
    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={'age': age})
    if fit:
        m.fit(data)
    return data, m

def create_allbus_model_2(filename='test_allbus.csv', modelname='allbus_model_2', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data = pd.read_csv(filename, index_col=0)
    data = data.drop(['eastwest', 'lived_abroad', 'spectrum', 'sex', 'educ', 'health'], axis=1)
    # Reduce size of data to improve performance
    data = data.sample(n=500, random_state=1)
    data.sort_index(inplace=True)
    # Set up shared variables
    age = theano.shared(np.array(data['age']))
    allbus_model = pm.Model()
    with allbus_model:
        # priors
        alpha_sd = pm.Uniform('alpha_sd', 0, 2000)
        beta_sd = pm.Uniform('beta_sd', 0, 1000)
        loc_transform = pm.Normal('loc_transform', 50, 20)
        scale_transform = pm.Uniform('scale_transform', 0, 100)
        sd_happ = pm.Uniform('sd_happ', 0, 5)
        alpha_happ1 = pm.Uniform('alpha_happ1', -10, 5)
        alpha_happ2 = pm.Uniform('alpha_happ2', 5, 10)
        beta_happ1 = pm.Uniform('beta_happ1', -0.001, 0.001)
        beta_happ2 = pm.Uniform('beta_happ2', -0.001, 0.001)
        # likelihood
        # transform age so that it resembles a bell-shaped distribution
        def normal_pdf(x,loc,scale):
            return 1/np.sqrt(2*math.pi*scale*scale)*np.exp(-(x-loc)**2/(2*scale*scale))
        age_transformed = normal_pdf(age, loc=loc_transform, scale=scale_transform)
        sd_income = alpha_sd + beta_sd*age_transformed
        income = pm.HalfNormal('income', sd_income, observed=data['income'])
        switchpoint = pm.Uniform('switchpoint', 500, 2000)
        beta_happ = pm.math.switch(income < switchpoint, beta_happ1, beta_happ2)
        alpha_happ = pm.math.switch(income < switchpoint, alpha_happ1, alpha_happ2)
        mu_happ = alpha_happ + beta_happ * income
        happiness = pm.Normal('happiness', mu_happ, sd_happ, observed=data['happiness'])

    # Create model instance for Lumen
    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={'age': age})
    if fit:
        m.fit(data)
    return data, m

def create_allbus_model_3(filename='test_allbus.csv', modelname='allbus_model_3', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data = pd.read_csv(filename, index_col=0)
    data = data.drop(['eastwest', 'lived_abroad', 'spectrum', 'sex', 'educ', 'health'], axis=1)
    # Reduce size of data to improve performance
    data = data.sample(n=500, random_state=1)
    data.sort_index(inplace=True)
    # Set up shared variables
    age = theano.shared(np.array(data['age']))
    allbus_model = pm.Model()
    with allbus_model:
        # priors
        alpha_sd = pm.Uniform('alpha_sd', 0, 2000)
        beta_sd = pm.Uniform('beta_sd', 0, 2000)
        loc_transform = pm.Normal('loc_transform', 50, 20)
        scale_transform = pm.Uniform('scale_transform', 0, 100)
        mu_happ = pm.Uniform('mu_happ', 6, 10)
        sd_happ1 = pm.Uniform('sd_happ1', 0, 8)
        sd_happ2 = pm.Uniform('sd_happ2', 0, 2)
        # likelihood
        # transform age so that it resembles a bell-shaped distribution
        def normal_pdf(x,loc,scale):
            return 1/np.sqrt(2*math.pi*scale*scale)*np.exp(-(x-loc)**2/(2*scale*scale))
        age_transformed = normal_pdf(age, loc=loc_transform, scale=scale_transform)
        sd_income = alpha_sd + beta_sd*age_transformed
        income = pm.HalfNormal('income', sd_income, observed=data['income'])
        switchpoint = pm.Uniform('switchpoint', 2000, 6000)
        sd_happ = pm.math.switch(income < switchpoint, sd_happ1, sd_happ2)
        happiness = pm.Normal('happiness', mu_happ, sd_happ, observed=data['happiness'])

    # Create model instance for Lumen
    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={'age': age})
    if fit:
        m.fit(data)
    return data, m

def create_allbus_model_4(filename='test_allbus.csv', modelname='allbus_model_4', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data = pd.read_csv(filename, index_col=0)
    data = data.drop(['eastwest', 'lived_abroad', 'spectrum', 'sex', 'educ', 'health'], axis=1)
    # Reduce size of data to improve performance
    data = data.sample(n=500, random_state=1)
    data.sort_index(inplace=True)
    # Set up shared variables
    age = theano.shared(np.array(data['age']))
    allbus_model = pm.Model()
    with allbus_model:
        # priors
        alpha_sd = pm.Uniform('alpha_sd', 0, 2000)
        beta_sd = pm.Uniform('beta_sd', 0, 1000)
        loc_transform = pm.Normal('loc_transform', 50, 20)
        scale_transform = pm.Uniform('scale_transform', 0, 100)
        sd_happ = pm.Uniform('sd_happ', 0, 5)
        alpha_happ1 = pm.Uniform('alpha_happ1', -10, 5)
        alpha_happ2 = pm.Uniform('alpha_happ2', 5, 10)
        beta_happ1 = pm.Uniform('beta_happ1', -0.001, 0.001)
        beta_happ2 = pm.Uniform('beta_happ2', -0.001, 0.001)
        # likelihood
        # transform age so that it resembles a bell-shaped distribution
        def normal_pdf(x,loc,scale):
            return 1/np.sqrt(2*math.pi*scale*scale)*np.exp(-(x-loc)**2/(2*scale*scale))
        age_transformed = normal_pdf(age, loc=loc_transform, scale=scale_transform)
        sd_income = alpha_sd + beta_sd*age_transformed
        income = pm.HalfNormal('income', sd_income, observed=data['income'])
        switchpoint = pm.Uniform('switchpoint', 500, 2000)
        beta_happ = pm.math.switch(income < switchpoint, beta_happ1, beta_happ2)
        alpha_happ = pm.math.switch(income < switchpoint, alpha_happ1, alpha_happ2)
        mu_happ = alpha_happ + beta_happ * income
        # Cap happiness at 10
        mu_happ = pm.math.switch(mu_happ < 10, mu_happ, 10)
        happiness = pm.Normal('happiness', mu_happ, sd_happ, observed=data['happiness'])

    # Create model instance for Lumen
    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={'age': age})
    if fit:
        m.fit(data)
    return data, m

def create_allbus_model_5(filename='test_allbus.csv', modelname='allbus_model_5', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data = pd.read_csv(filename, index_col=0)
    data = data.drop(['eastwest', 'lived_abroad', 'spectrum', 'sex', 'educ', 'health'], axis=1)
    # Reduce size of data to improve performance
    data = data.sample(n=500, random_state=1)
    data.sort_index(inplace=True)
    # Set up shared variables
    age = theano.shared(np.array(data['age']))
    allbus_model = pm.Model()
    with allbus_model:
        # priors
        alpha_sd = pm.Uniform('alpha_sd', 0, 2000)
        beta_sd = pm.Uniform('beta_sd', 0, 1000)
        loc_transform = pm.Normal('loc_transform', 50, 20)
        scale_transform = pm.Uniform('scale_transform', 0, 100)
        sd_happ = pm.Uniform('sd_happ', 0, 5)
        k_logistic = pm.Uniform('k_logistic', 0, 10)
        x_0_logistic = pm.Uniform('x_0_logistic', 0, 1000)
        # likelihood
        # transform age so that it resembles a bell-shaped distribution
        def normal_pdf(x,loc,scale):
            return 1/np.sqrt(2*math.pi*scale**2)*np.exp(-(x-loc)**2/(2*scale**2))
        age_transformed = normal_pdf(age, loc=loc_transform, scale=scale_transform)
        sd_income = alpha_sd + beta_sd*age_transformed
        age + loc_transform
        income = pm.HalfNormal('income', sd_income, observed=data['income'])

        def logistic_func(l, k, x, x_0):
            # l: curve's max value
            # x_0: x_value of sigmoid's mid-point
            # k: steepness of the curve
            return l/(1+np.exp(-k*(x-x_0)))

        mu_happ = logistic_func(10, k_logistic, income, x_0_logistic)

        happiness = pm.Normal('happiness', mu_happ, sd_happ, observed=data['happiness'])

    # Create model instance for Lumen
    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={'age': age})
    if fit:
        m.fit(data)
    return data, m

######################################
# Call all model generating functions
######################################
if __name__ == '__main__':

    start = timeit.default_timer()

    try:
        testcasemodel_path = user_cfg['modules']['modelbase']['test_model_directory']
        testcasedata_path = user_cfg['modules']['modelbase']['test_data_directory']
    except KeyError:
        print('Specify a test_model_directory and a test_data_direcory in run_conf.py')
        raise

    # This list specifies which models are created when the script is run. If you only want to create
    # specific models, adjust the list accordingly
    create_functions = [create_pymc3_simplest_model, create_pymc3_getting_started_model,
                        create_pymc3_getting_started_model_independent_vars,
                        create_pymc3_coal_mining_disaster_model,
                        create_getting_started_model_shape, create_flight_delay_model_1, create_flight_delay_model_2,
                        create_flight_delay_model_3, create_allbus_model_4]

    create_functions = [create_flight_delay_model_1, create_flight_delay_model_2,
                        create_flight_delay_model_3]

    for func in create_functions:
        data, m = func(fit=False)
        data, m_fitted = func(fit=True)

        # create empirical model
        name = "emp_" + m.name
        m.set_empirical_model_name(name)
        m_fitted.set_empirical_model_name(name)
        emp_model = EmpiricalModel(name=name)
        emp_model.fit(df=data)

        m_fitted.save(testcasemodel_path)
        m.save(testcasemodel_path)
        emp_model.save(testcasemodel_path)

        data.to_csv(os.path.join(testcasedata_path, m.name + '.csv'), index=False)

    stop = timeit.default_timer()
    print('Time: ', stop - start)