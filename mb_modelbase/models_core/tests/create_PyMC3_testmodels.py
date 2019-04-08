import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model

testcasemodel_path = '/home/guet_jn/Desktop/mb_data/data_models/'
# testcasemodel_path = '/home/philipp/Documents/projects/graphical_models/code/mb_data/data_models/'

######################################
# pymc3_testcase_model
#####################################

# modelname = 'pymc3_testcase_model'
# np.random.seed(2)
# size = 100
# mu = np.random.normal(0, 1, size=size)
# sigma = 1
# X = np.random.normal(mu, sigma, size=size)
# data = pd.DataFrame({'X': X})
#
# basic_model = pm.Model()
# with basic_model:
#     sigma = 1
#     mu = pm.Normal('mu', mu=0, sd=sigma)
#     X = pm.Normal('X', mu=mu, sd=sigma, observed=data['X'])
#
#     nr_of_samples = 10000
#     trace = pm.sample(nr_of_samples, tune=1000, cores=4)
# m = ProbabilisticPymc3Model(modelname, basic_model)
# Model.save(m, testcasemodel_path + modelname + '.mdl')

######################################
# pymc3_getting_started_model
######################################

# modelname = 'pymc3_getting_started_model_fitted'
# np.random.seed(123)
# alpha, sigma = 1, 1
# beta_0 = 1
# beta_1 = 2.5
# size = 100
# X1 = np.random.randn(size)
# X2 = np.random.randn(size) * 0.2
# Y = alpha + beta_0 * X1 + beta_1 * X2 + np.random.randn(size) * sigma
# data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
#
#
# basic_model = pm.Model()
#
# with basic_model:
#     # Priors for unknown model parameters
#     alpha = pm.Normal('alpha', mu=0, sd=10)
#     beta_0 = pm.Normal('beta_0', mu=0, sd=10)
#     beta_1 = pm.Normal('beta_1', mu=0, sd=10)
#     sigma = pm.HalfNormal('sigma', sd=1)
#
#     # Expected value of outcome
#     mu = alpha + beta_0 * data['X1'] + beta_1 * data['X2']
#
#     # Likelihood (sampling distribution) of observations
#     Y = pm.Normal('Y', mu=mu, sd=sigma, observed=data['Y'])
#     X1 = pm.Normal('X1', mu=data['X1'], sd=sigma, observed=data['X1'])
#     X2 = pm.Normal('X2', mu=data['X2'], sd=sigma, observed=data['X2'])
#
# m = ProbabilisticPymc3Model(modelname, basic_model)
# m.fit(data)
# Model.save(m, testcasemodel_path + modelname + '.mdl')

######################################
# pymc3_coal_mining_disaster_model
######################################

modelname = 'pymc3_coal_mining_disaster_model'

disasters = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 3, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
years = np.arange(1851, 1962)

data = pd.DataFrame({'years': years, 'disasters': disasters})
with pm.Model() as disaster_model:

    switchpoint = pm.DiscreteUniform('switchpoint', lower=years.min(), upper=years.max(), testval=1900)

    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential('early_rate', 1)
    late_rate = pm.Exponential('late_rate', 1)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

    disasters = pm.Poisson('disasters', rate, observed=data['disasters'])
    years = pm.Normal('years', mu=data['years'], sd=0.1, observed=data['years'])

m = ProbabilisticPymc3Model(modelname, disaster_model)
#m.fit(data)
Model.save(m, testcasemodel_path + modelname + '.mdl')