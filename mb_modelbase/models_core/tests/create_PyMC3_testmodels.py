import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
import theano
import pickle
import matplotlib.pyplot as plt
from pylab import hist

testcasemodel_path = '/home/guet_jn/Desktop/mb_data/data_models/'
testcasedata_path = '/home/guet_jn/Desktop/mb_data/mb_data/pymc3/'

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
# m.fit(data)
# Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')

######################################
# pymc3_getting_started_model
######################################

# modelname = 'pymc3_getting_started_model'
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
# Model.save(m, testcasemodel_path + modelname + '.mdl')
# m.fit(data)
# Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')
###############################################
# pymc3_getting_started_model_independent vars
###############################################

modelname = 'pymc3_getting_started_model_independent_vars'
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
Model.save(m, testcasemodel_path + modelname + '.mdl')
m.fit(data)
Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')
data.to_csv(testcasedata_path + modelname + '.csv', index=False)
# #pickle.dump([X1, X2], open(testcasedata_path + modelname + '_shared_vars.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
######################################
# pymc3_coal_mining_disaster_model
######################################

# modelname = 'pymc3_coal_mining_disaster_model'
#
# disasters = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
#                             3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
#                             2, 2, 3, 4, 2, 1, 3, 3, 2, 1, 1, 1, 1, 3, 0, 0,
#                             1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
#                             0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
#                             3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
#                             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
# years = np.arange(1851, 1962)
#
# data = pd.DataFrame({'years': years, 'disasters': disasters})
# years = theano.shared(years)
# with pm.Model() as disaster_model:
#
#     switchpoint = pm.DiscreteUniform('switchpoint', lower=years.get_value().min(), upper=years.get_value().max(),
#     testval=1900)
#
#     # Priors for pre- and post-switch rates number of disasters
#     early_rate = pm.Exponential('early_rate', 1)
#     late_rate = pm.Exponential('late_rate', 1)
#
#     # Allocate appropriate Poisson rates to years before and after current
#     rate = pm.math.switch(switchpoint >= years.get_value(), early_rate, late_rate)
#
#     disasters = pm.Poisson('disasters', rate, observed=data['disasters'])
#     #years = pm.Normal('years', mu=data['years'], sd=0.1, observed=data['years'])
#
# m = ProbabilisticPymc3Model(modelname, disaster_model,shared_vars={'years':years})
# Model.save(m, testcasemodel_path + modelname + '.mdl')
# m.fit(data)
# Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')
######################################
# eight_schools_model
######################################

# modelname = 'eight_schools_model'
#
# scores = [28.39,7.94,-2.75,6.82,-0.64,0.63,18.01,12.16]
# standard_errors = [14.9,10.2,16.3,11.0,9.4,11.4,10.4,17.6]
# data = pd.DataFrame({'test_scores': scores, 'standard_errors': standard_errors})
# standard_errors = theano.shared(standard_errors)
#
# with pm.Model() as normal_normal_model:
#     tau = pm.Uniform('tau',lower=0,upper=10)
#     mu = pm.Uniform('mu',lower=0,upper=10)
#     theta_1 = pm.Normal('theta_1', mu=mu, sd=tau)
#     theta_2 = pm.Normal('theta_2', mu=mu, sd=tau)
#     theta_3 = pm.Normal('theta_3', mu=mu, sd=tau)
#     theta_4 = pm.Normal('theta_4', mu=mu, sd=tau)
#     theta_5 = pm.Normal('theta_5', mu=mu, sd=tau)
#     theta_6 = pm.Normal('theta_6', mu=mu, sd=tau)
#     theta_7 = pm.Normal('theta_7', mu=mu, sd=tau)
#     theta_8 = pm.Normal('theta_8', mu=mu, sd=tau)
#
#     test_scores = pm.Normal('test_scores',
#                             mu=[theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8],
#                             sd=standard_errors.get_value(), observed=data['test_scores'])

#    trace = pm.sample(1000,chains=1,cores=1)
#    simulated_scores = np.asarray(pm.sample_ppc(trace)[str("test_scores")])
#
# # Compute test statistics
# disc_mean = [np.mean(simvals) for simvals in simulated_scores]
# disc_min = [min(simvals) for simvals in simulated_scores]
# disc_max = [max(simvals) for simvals in simulated_scores]
# disc_std = [np.std(simvals) for simvals in simulated_scores]
#
# # Plot test statistics
# vis_grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)
# plt.subplot(vis_grid[0, 0])
# hist(disc_mean,bins=15,edgecolor='black',color='grey')
# plt.title('mean')
# plt.subplot(vis_grid[1, 0])
# hist(disc_min,bins=15,edgecolor='black',color='grey')
# plt.title('min')
# plt.subplot(vis_grid[1, 1])
# hist(disc_max,bins=15,edgecolor='black',color='grey')
# plt.title('max')
# plt.subplot(vis_grid[0, 1])
# hist(disc_std,bins=15,edgecolor='black',color='grey')
# plt.title('standard deviation')

# m = ProbabilisticPymc3Model(modelname, normal_normal_model, shared_vars={'standard_errors': standard_errors})
# Model.save(m, testcasemodel_path + modelname + '.mdl')
# m.fit(data)
# Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')

######################################
# more_than_eight_schools_model
######################################

# modelname = 'more_schools_model_fitted'
#
# a1 = np.random.normal(loc=28.39,size=6)
# a2 = np.random.normal(loc=7.94,size=6)
# a3 = np.random.normal(loc=-2.75,size=6)
# a4 = np.random.normal(loc=6.82,size=6)
# a5 = np.random.normal(loc=-0.64,size=6)
# a6 = np.random.normal(loc=0.63,size=6)
# a7 = np.random.normal(loc=18.01,size=6)
# a8 = np.random.normal(loc=12.16,size=6)
# scores = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8))
#
# a1 = np.random.normal(loc=14.9,size=6)
# a2 = np.random.normal(loc=10.2,size=6)
# a3 = np.random.normal(loc=16.3,size=6)
# a4 = np.random.normal(loc=11.0,size=6)
# a5 = np.random.normal(loc=9.4,size=6)
# a6 = np.random.normal(loc=11.4,size=6)
# a7 = np.random.normal(loc=10.4,size=6)
# a8 = np.random.normal(loc=17.6,size=6)
# standard_errors = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8))
#
# # Shuffle data
# indices = np.arange(48)
# np.random.shuffle(indices)
# scores = scores[indices]
# standard_errors = standard_errors[indices]
#
# data = pd.DataFrame({'test_scores': scores, 'standard_errors': standard_errors})
#
# with pm.Model() as normal_normal_model:
#     tau = pm.Uniform('tau',lower=0,upper=10)
#     mu = pm.Uniform('mu',lower=0,upper=10)
#     theta_1 = pm.Normal('theta_1', mu=mu, sd=tau)
#     theta_2 = pm.Normal('theta_2', mu=mu, sd=tau)
#     theta_3 = pm.Normal('theta_3', mu=mu, sd=tau)
#     theta_4 = pm.Normal('theta_4', mu=mu, sd=tau)
#     theta_5 = pm.Normal('theta_5', mu=mu, sd=tau)
#     theta_6 = pm.Normal('theta_6', mu=mu, sd=tau)
#     theta_7 = pm.Normal('theta_7', mu=mu, sd=tau)
#     theta_8 = pm.Normal('theta_8', mu=mu, sd=tau)
#     theta_9 = pm.Normal('theta_9', mu=mu, sd=tau)
#     theta_10 = pm.Normal('theta_10', mu=mu, sd=tau)
#     theta_11 = pm.Normal('theta_11', mu=mu, sd=tau)
#     theta_12 = pm.Normal('theta_12', mu=mu, sd=tau)
#     theta_13 = pm.Normal('theta_13', mu=mu, sd=tau)
#     theta_14 = pm.Normal('theta_14', mu=mu, sd=tau)
#     theta_15 = pm.Normal('theta_15', mu=mu, sd=tau)
#     theta_16 = pm.Normal('theta_16', mu=mu, sd=tau)
#     theta_17 = pm.Normal('theta_17', mu=mu, sd=tau)
#     theta_18 = pm.Normal('theta_18', mu=mu, sd=tau)
#     theta_19 = pm.Normal('theta_19', mu=mu, sd=tau)
#     theta_20 = pm.Normal('theta_20', mu=mu, sd=tau)
#     theta_21 = pm.Normal('theta_21', mu=mu, sd=tau)
#     theta_22 = pm.Normal('theta_22', mu=mu, sd=tau)
#     theta_23 = pm.Normal('theta_23', mu=mu, sd=tau)
#     theta_24 = pm.Normal('theta_24', mu=mu, sd=tau)
#     theta_25 = pm.Normal('theta_25', mu=mu, sd=tau)
#     theta_26 = pm.Normal('theta_26', mu=mu, sd=tau)
#     theta_27 = pm.Normal('theta_27', mu=mu, sd=tau)
#     theta_28 = pm.Normal('theta_28', mu=mu, sd=tau)
#     theta_29 = pm.Normal('theta_29', mu=mu, sd=tau)
#     theta_30 = pm.Normal('theta_30', mu=mu, sd=tau)
#     theta_31 = pm.Normal('theta_31', mu=mu, sd=tau)
#     theta_32 = pm.Normal('theta_32', mu=mu, sd=tau)
#     theta_33 = pm.Normal('theta_33', mu=mu, sd=tau)
#     theta_34 = pm.Normal('theta_34', mu=mu, sd=tau)
#     theta_35 = pm.Normal('theta_35', mu=mu, sd=tau)
#     theta_36 = pm.Normal('theta_36', mu=mu, sd=tau)
#     theta_37 = pm.Normal('theta_37', mu=mu, sd=tau)
#     theta_38 = pm.Normal('theta_38', mu=mu, sd=tau)
#     theta_39 = pm.Normal('theta_39', mu=mu, sd=tau)
#     theta_40 = pm.Normal('theta_40', mu=mu, sd=tau)
#     theta_41 = pm.Normal('theta_41', mu=mu, sd=tau)
#     theta_42 = pm.Normal('theta_42', mu=mu, sd=tau)
#     theta_43 = pm.Normal('theta_43', mu=mu, sd=tau)
#     theta_44 = pm.Normal('theta_44', mu=mu, sd=tau)
#     theta_45 = pm.Normal('theta_45', mu=mu, sd=tau)
#     theta_46 = pm.Normal('theta_46', mu=mu, sd=tau)
#     theta_47 = pm.Normal('theta_47', mu=mu, sd=tau)
#     theta_48 = pm.Normal('theta_48', mu=mu, sd=tau)
#
#     theta = [theta_1,theta_2,theta_3,theta_4,theta_5,theta_6,theta_7,theta_8,theta_9,theta_10,theta_11,theta_12,
#     theta_13,theta_14,theta_15,theta_16,theta_17,theta_18,theta_19,theta_20,theta_21,theta_22,theta_23,theta_24,
#     theta_25,theta_26,theta_27,theta_28,theta_29,theta_30,theta_31,theta_32,theta_33,theta_34,theta_35,theta_36,
#     theta_37,theta_38,theta_39,theta_40,theta_41,theta_42,theta_43,theta_44,theta_45,theta_46,theta_47,theta_48]
#
#     test_scores = pm.Normal('test_scores',mu=theta,sd=data['standard_errors'], observed=data['test_scores'])
#
#     m = ProbabilisticPymc3Model(modelname, normal_normal_model)
#     Model.save(m, testcasemodel_path + modelname + '.mdl')
#     m.fit(data)
#     Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')
#
######################################
# eight_schools_model_shape
######################################

# modelname = 'eight_schools_model_shape'
#
# scores = [28.39,7.94,-2.75,6.82,-0.64,0.63,18.01,12.16]
# standard_errors = [14.9,10.2,16.3,11.0,9.4,11.4,10.4,17.6]
# data = pd.DataFrame({'test_scores': scores, 'standard_errors': standard_errors})
#
# with pm.Model() as normal_normal_model:
#     tau = pm.Uniform('tau',lower=0,upper=10)
#     mu = pm.Uniform('mu',lower=0,upper=10)
#     theta = pm.Normal('theta', mu=mu, sd=tau, shape=8)
#     test_scores = pm.Normal('test_scores',mu=theta,sd=data['standard_errors'], observed=data['test_scores'])
#
# m = ProbabilisticPymc3Model(modelname, normal_normal_model)
# Model.save(m, testcasemodel_path + modelname)
# m.fit(data)
# Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')