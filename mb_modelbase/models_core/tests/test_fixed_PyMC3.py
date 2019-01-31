import numpy as np
import pandas as pd
import pymc3 as pm
import mb_modelbase as mbase


# # Generate data
# np.random.seed(2)
# size = 100
# mu = np.random.normal(0, 1, size=size)
# sigma = 1
# X = np.random.normal(mu, sigma, size=size)
# data = pd.DataFrame({'X': X})
#
# # Specify model
# basic_model = pm.Model()
# with basic_model:
#     sigma = 1
#     mu = pm.Normal('mu', mu=0, sd=sigma)
#     X = pm.Normal('X', mu=mu, sd=sigma, observed=data['X'])
#
#     nr_of_samples = 2000
#     trace = pm.sample(nr_of_samples, tune=1000, cores=4)
#
# modelname = 'my_pymc3_model'
# mymod = mbase.FixedProbabilisticModel(modelname, basic_model)
# mymod.fit(data)
#mymod = mbase.Model.load('/home/philipp/Documents/projects/graphical_models/code/mb_data/data_models/my_pymc3_model.mdl')
mymod = mbase.Model.load('/home/guet_jn/Desktop/mb_data/data_models/my_pymc3_model.mdl')
mymod.parallel_processing = False


mymod_2 = mymod.copy()
mymod_2 = mymod_2.condition([mbase.Condition("X", "<", -4)])
mymod_2 = mymod_2.condition([mbase.Condition("X", ">", -6)])
mymod_2.marginalize(remove=["X"])
res = mymod_2.aggregate("maximum")



#probabilitiy()


print(res)
