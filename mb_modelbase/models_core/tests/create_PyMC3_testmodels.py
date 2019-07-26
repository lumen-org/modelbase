import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
import theano
from run_conf import cfg as user_cfg

try:
    testcasemodel_path = user_cfg['modules']['modelbase']['test_model_directory'] + '/'
    testcasedata_path = user_cfg['modules']['modelbase']['test_data_directory'] + '/'
    #testcasemodel_path = '/home/luca_ph/Documents/projects/graphical_models/code/ppl_models/'
    #testcasedata_path = '/home/luca_ph/Documents/projects/graphical_models/code/ppl_models/'
except KeyError:
    print('Specify a test_model_directory and a test_data_direcory in run_conf.py')
    raise

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
    return m

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
        return m

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
    data.to_csv(testcasedata_path + modelname + '.csv', index=False)
    if fit:
        m.fit(data)
    return m

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
        data.to_csv(testcasedata_path + modelname + '.csv', index=False)
        m = ProbabilisticPymc3Model(modelname, basic_model, shared_vars={'X1': X1})
        if fit:
            m.fit(data)
        return m

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
    return m

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
    return m

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
    data.to_csv(testcasedata_path + modelname + '.csv', index=False)
    if fit:
        m.fit(data)
    return m

######################################
# Call all model generating functions
######################################
create_functions = [create_pymc3_simplest_model, create_pymc3_getting_started_model,
                    create_pymc3_getting_started_model_independent_vars,
                    create_pymc3_getting_started_model_independent_vars_nosharedvars,
                    create_pymc3_coal_mining_disaster_model, create_pymc3_eight_schools_model,
                    create_getting_started_model_shape]

for func in create_functions:
    m = func(fit=False)
    m_fitted = func(fit=True)
    Model.save(m, testcasemodel_path)
    Model.save(m_fitted, testcasemodel_path)