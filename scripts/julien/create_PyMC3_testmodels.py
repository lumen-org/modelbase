import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.models_core.empirical_model import EmpiricalModel
import theano
# from scripts.run_conf import cfg as user_cfg
import os
import timeit
import scipy.stats
import math

#import pymc_models_allbus
import pymc_models_titanic
import pymc_models_bank
#import pymc_models_burglary


######################################
# function template
#####################################
def create_example_model(modelname='my_name', fit=True):
    if fit:
        modelname = modelname + '_fitted'
    ## Load data as pandas df
    # data = pd.read_csv(...)
    example_model = pm.Model()
    ## Specify  your model
    # with example_model:
    # ...
    m = ProbabilisticPymc3Model(modelname, example_model)
    if fit:
        m.fit(data)
    return data, m


######################################
# pymc3_testcase_model
#####################################
def create_pymc3_simplest_model(modelname='pymc3_simplest_model', fit=True):
    if fit:
        modelname = modelname + '_fitted'
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
# Call all model generating functions
######################################
if __name__ == '__main__':
    modeldir = 'models'  # sub directory where to store created models
    mypath = "/home/julien/Development/nips2020/fitted_models/"
    if not os.path.exists(mypath):
        os.makedirs(mypath)
    start = timeit.default_timer()
    testcasemodel_path = mypath
    testcasedata_path = mypath

    from inspect import getmembers, isfunction
    import scripts.julien.pymc_models_bank2 as bank
    import scripts.julien.pymc_models_allbus2 as allbus

    module = allbus

    functions = [o[1] for o in getmembers(module) if isfunction(o[1])]
    #print(functions)

    # This list specifies which models are created when the script is run. If you only want to create
    # specific models, adjust the list accordingly
    create_functions = [
        # pymc_models_burglary.create_burglary_model

        #pymc_models_allbus.create_allbus_model_1
        #pymc_models_allbus.create_allbus_model_2
        #pymc_models_allbus.create_allbus_model_3
        #pymc_models_allbus.create_allbus_model_4,
        #pymc_models_allbus.create_allbus_model_5,
        #pymc_models_allbus.create_allbus_model_6
        #pymc_models_allbus.create_allbus_model_7
        #pymc_models_allbus.create_allbus_model_8

        #pymc_models_titanic.create_titanic_model_1,
        #pymc_models_titanic.create_titanic_model_2,
        #pymc_models_titanic.create_titanic_model_3,
        #pymc_models_titanic.create_titanic_model_4,
        #pymc_models_titanic.create_titanic_model_5,

        #pymc_models_bank.create_bank_model_1,
    ]
    create_functions = functions

    for func in create_functions:
        data, m_fitted = func(fit=True)

        # create empirical model
        name = "emp_" + m_fitted.name
        m_fitted.set_empirical_model_name(name)
        emp_model = EmpiricalModel(name=name)
        emp_model.fit(df=data)

        m_fitted.save(testcasemodel_path)
        print(f"Saved in {testcasemodel_path}")
        emp_model.save(testcasemodel_path)
        if data is not None:
            data.to_csv(os.path.join(testcasedata_path, m_fitted.name + '.csv'), index=False)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
