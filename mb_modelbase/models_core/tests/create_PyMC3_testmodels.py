import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
import theano
#from run_conf import cfg as user_cfg
import os

try:
    #testcasemodel_path = user_cfg['modules']['modelbase']['test_model_directory'] + '/'
    #testcasedata_path = user_cfg['modules']['modelbase']['test_data_directory'] + '/'
    # testcasemodel_path = '/home/luca_ph/Documents/projects/graphical_models/code/ppl_models/'
    # testcasedata_path = '/home/luca_ph/Documents/projects/graphical_models/code/ppl_models/'
    testcasemodel_path = '.'
    testcasedata_path = '.'

except KeyError:
    print('Specify a test_model_directory and a test_data_direcory in run_conf.py')
    raise

# TODO: refactor to coding style like in testmodels_pymc3.py
#  --> make it reusable and modular

######################################
# pymc3_testcase_model
#####################################
modelname = 'pymc3_simplest_model'
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
Model.save(m, testcasemodel_path)
m = ProbabilisticPymc3Model(modelname + '_fitted', basic_model)
m.fit(data)
Model.save(m, testcasemodel_path)
######################################
# pymc3_getting_started_model
######################################

modelname = 'pymc3_getting_started_model'
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
Model.save(m, testcasemodel_path)
m = ProbabilisticPymc3Model(modelname + '_fitted', basic_model)
m.fit(data)
Model.save(m, testcasemodel_path)
##############################################
# pymc3_getting_started_model_independent vars
##############################################

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
Model.save(m, testcasemodel_path)
m = ProbabilisticPymc3Model(modelname + '_fitted', basic_model, shared_vars={'X1': X1, 'X2': X2})
m.fit(data)
Model.save(m, testcasemodel_path)
data.to_csv(testcasedata_path + modelname + '.csv', index=False)
###########################################################
# pymc3_getting_started_model_independent vars_nosharedvars
###########################################################

modelname = 'pymc3_getting_started_model_independent_vars_nosharedvars'
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
Model.save(m, testcasemodel_path)
m = ProbabilisticPymc3Model(modelname + '_fitted', basic_model, shared_vars={'X1': X1})
data.to_csv(testcasedata_path + modelname + '.csv', index=False)
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
Model.save(m, testcasemodel_path)
m = ProbabilisticPymc3Model(modelname + '_fitted', disaster_model, shared_vars={'years': years})
m.fit(data)
Model.save(m, testcasemodel_path)

########################################
# eight_schools_model
########################################

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

m = ProbabilisticPymc3Model(modelname, normal_normal_model,
                            shared_vars={'standard_errors': standard_errors}, fixed_data_length=True)
Model.save(m, testcasemodel_path)
m = ProbabilisticPymc3Model(modelname + '_fitted', normal_normal_model,
                            shared_vars={'standard_errors': standard_errors}, fixed_data_length=True)
m.fit(data)
Model.save(m, testcasemodel_path)

######################################
# getting_started_model_shape
######################################
modelname = 'pymc3_getting_started_model_shape'
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
Model.save(m, testcasemodel_path)
m = ProbabilisticPymc3Model(modelname + '_fitted', basic_model, shared_vars={'X1': X1, 'X2': X2})
m.fit(data)
Model.save(m, testcasemodel_path)
data.to_csv(testcasedata_path + modelname + '.csv', index=False)

######################################
# Chagos rats d15N
######################################

modelname = 'Chagos Rats d15N'
# Return list of unique items and an index of their position in L
def indexall(L):
    poo = []
    for p in L:
        if not p in poo:
            poo.append(p)
    Ix = np.array([poo.index(p) for p in L])
    return poo,Ix

# Return list of unique items and an index of their position in long, relative to short
def subindexall(short,long):
    poo = []
    out = []
    for s,l in zip(short,long):
        if not l in poo:
            poo.append(l)
            out.append(s)
    return indexall(out)

# Function to standardize covariates
def stdize(x):
    return (x-np.mean(x))/(2*np.std(x))

# Import age and growth data
path = testcasedata_path + 'Chagos_isotope_data.csv'
if not os.path.isfile(path):
    print('Training data for Chagos rats not found. Store the training data in ' + testcasedata_path)
xdata = pd.read_csv(path)

# Make arrays locally available
organism = xdata.Tissue.values
Organism,Io = indexall(organism)
norg = len(Organism)

It = xdata.Treatment.values=='Rats'

atoll = xdata.Atoll.values
island = xdata.Island.values

Atoll,Ia = subindexall(atoll,island)
natoll = len(Atoll)
Island,Is = indexall(island)
nisland = len(Island)
TNI = np.log(np.array([xdata.TNI_reef_area[list(island).index(i)] for i in Island]))
ReefArea = np.log(np.array([xdata.Reef_area[list(island).index(i)] for i in Island]))

# Distance to shore in metres
Dshore_ = xdata.To_shore_m.values
Dshore_[np.isnan(Dshore_)] = 0
Dshore = stdize(Dshore_)
Dshore[Dshore<0] = 0

d15N = xdata.cal_d15N.values

basic_model = pm.Model()
It = It.astype(int)

data = pd.DataFrame({'Is': Is, 'Io': Io, 'It': It, 'Dshore': Dshore, 'Yi': d15N})
Is = theano.shared(Is)
It = theano.shared(It)
Io = theano.shared(Io)
Dshore = theano.shared(Dshore)

with basic_model:
    # Global prior
    γ0 = pm.Normal('Mean_d15N', mu=0.0, tau=0.001)
    # Reef-area effect
    # γ1 = pm.Normal('ReefArea', mu=0.0, tau=0.001)

    # Island-level model
    # γ = γ0+γ1*ReefArea
    γ = γ0
    # Inter-island variablity
    σγ = pm.Uniform('SD_reef', lower=0, upper=100)
    τγ = σγ ** -2
    β0 = pm.Normal('Island_d15N', mu=γ, tau=τγ, shape=nisland)

    # Organism mean (no rats)
    β1_ = pm.Normal('Organism_', mu=0.0, tau=0.001, shape=norg - 1)
    β1 = theano.tensor.set_subtensor(theano.tensor.zeros(shape=norg)[1:], β1_)

    # Organism-specific rat effects
    β2 = pm.Normal('Rat_effect_', mu=0.0, tau=0.001, shape=norg)

    # Distance to shore
    β3 = pm.Normal('Dist_to_shore', mu=0.0, tau=0.001)

    # Mean model
    μ = β0[Is] + β1[Io] + β2[Io] * It + β3 * Dshore

    # Organism-specific variance
    σ = pm.Uniform('SD', lower=0, upper=100)
    τ = σ ** -2

    # Likelihood
    Yi = pm.StudentT('Yi', nu=4, mu=μ, lam=τ, observed=d15N)

m = ProbabilisticPymc3Model(modelname, basic_model, shared_vars={'Is': Is, 'Io': Io, 'It': It, 'Dshore': Dshore})
Model.save(m, testcasemodel_path + modelname + '.mdl')
m = ProbabilisticPymc3Model(modelname + '_fitted', basic_model, shared_vars={'Is': Is, 'Io': Io, 'It': It, 'Dshore': Dshore})
m.fit(data)
Model.save(m, testcasemodel_path + modelname + '_fitted.mdl')

######################################
# Chagos rats vonB
######################################

modelname = 'Chagos Rats vonB'

# Helper functions
def indexall(L):
    poo = []
    for p in L:
        if not p in poo:
            poo.append(p)
    Ix = np.array([poo.index(p) for p in L])
    return poo,Ix

def subindexall(short,long):
    poo = []
    out = []
    for s,l in zip(short,long):
        if not l in poo:
            poo.append(l)
            out.append(s)
    return indexall(out)

match = lambda a, b: np.array([ b.index(x) if x in b else None for x in a ])

path = testcasedata_path + 'chagos_otolith.csv'
if not os.path.isfile(path):
    print('Training data for Chagos rats not found. Store the training data in ' + testcasedata_path)
# Import age and growth data
xdata = pd.read_csv(path)

# Site
site = xdata.Site.values

# Fish ID
ID = xdata.OtolithID.values

# Length
TL = xdata.TL.values
lTL = np.log(TL)
maxTL = max(TL)
minTL = min(TL)

# Bird or rat island
Treatment,It = indexall(xdata.Treatment.values)

# Age
age = xdata.Age.values

# Plotting age
agex = np.linspace(min(age),max(age),num=100)

Model = pm.Model()

with Model:
    Linf = pm.Uniform('Linf',maxTL, maxTL*2)
    L0 = pm.Uniform('L0', 0, minTL)
    k0 = pm.Uniform('k0', 0.001, 1)
    k1 = pm.Normal('k1', 0, 10)
    σ = pm.Uniform('σ', 0, 1000)
    μ = theano.tensor.log(Linf-(Linf-L0)*theano.tensor.exp(-(k0+k1*It)*age))
    yi = pm.Normal('yi',μ, σ, observed=lTL)