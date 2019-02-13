import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.fixed_PyMC3_model import FixedProbabilisticModel

testcasemodel_path = '/home/guet_jn/Desktop/mb_data/data_models/'
# testcasemodel_path = '/home/philipp/Documents/projects/graphical_models/code/mb_data/data_models/'

######################################
# pymc3_testcase_model
#####################################

modelname = 'pymc3_testcase_model'
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

    nr_of_samples = 10000
    trace = pm.sample(nr_of_samples, tune=1000, cores=4)
m = FixedProbabilisticModel(modelname, basic_model)
Model.save(m, testcasemodel_path + modelname + '.mdl')

######################################
# pymc3_getting_started_model
#####################################

modelname = 'pymc3_getting_started_model'
np.random.seed(123)
alpha, sigma = 1, 1
beta = [1, 2.5]
size = 100
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2
Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

m = FixedProbabilisticModel(modelname, basic_model)
m.fit(data)
Model.save(m, testcasemodel_path + modelname + '.mdl')
