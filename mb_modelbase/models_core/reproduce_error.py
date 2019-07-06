import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import pymc3 as pm
import theano

modelname = 'eight_schools_model'

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

    trace = pm.sample(10,chains=1,cores=1)
    ppc = pm.sample_ppc(trace)