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

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper
from mb_modelbase.utils.Metrics import cll

#import pymc_models_allbus
#import pymc_models_burglary

allbus_backward_map = {
    'eastwest' : {0: 'East', 1: 'West'},
    'sex' : {0 : 'Female', 1: 'Male'},
    'happiness': {0: 'h0', 1: 'h1', 2: 'h2', 3: 'h3', 4: 'h4', 5: 'h5', 6: 'h6', 7: 'h7', 8: 'h8', 9: 'h9', 10: 'h10'}
}
allbus_forward_map = {
    'eastwest' : {'East' : 0, 'West' : 1},
    'sex' : {'Female' : 0, 'Male' : 1},
    'happiness': {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8, 'h9': 9, 'h10': 10}
}
dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)

train_filename="/home/julien/PycharmProjects/modelbase/scripts/julien/data/allbus_h_train.csv"
test_filename="/home/julien/PycharmProjects/modelbase/scripts/julien/data/allbus_h_test.csv"

#test_data = pd.read_csv(test_filename)
#test_data = test_data.drop(['lived_abroad', 'health'], axis=1)
#test_data["happiness"] = pd.Series(test_data["happiness"]).map(allbus_forward_map["happiness"])

model_file = "/home/julien/PycharmProjects/modelbase/scripts/julien/allbus.dat"

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
# hand tuned model
#####################################
def create_hand_tuned_model(modelname='allbus_hand_tuned', fit=True):
    if fit:
        modelname = modelname+'_fitted'
    # Load and prepare data
    data = pd.read_csv(train_filename)
    data = data.drop(['lived_abroad', 'health'], axis=1)
    data["happiness"] = pd.Series(data["happiness"]).map(allbus_forward_map["happiness"])

    # Reduce size of data to improve performance
    data = data.sample(n=800, random_state=1)
    data.sort_index(inplace=True)

    # Set up shared variables
    age = theano.shared(np.array(data['age']))
    age_min = np.min(data['age'])
    age_max = np.max(data['age'])
    age_diff = age_max-age_min

    educ_diff = 4
    inc_max = np.max(data['income'])

    sex_transformed = [allbus_forward_map['sex'][x] for x in data['sex']]
    eastwest_transformed = [allbus_forward_map['eastwest'][x] for x in data['eastwest']]

    allbus_model = pm.Model()
    with allbus_model:
        age_mu = pm.Uniform('age_mu', age_min, age_max)
        age_sigma = pm.Uniform('age_sigma', 1, 50)
        age = pm.TruncatedNormal('age', mu=age_mu, sigma=age_sigma, lower=age_min, upper=age_max, observed=data['age'])

        educ_p = pm.Dirichlet('educ_p', np.ones(5), shape=5)
        educ = pm.Categorical('educ', p=educ_p, observed=data["educ"], shape=1)

        sex_p = pm.Dirichlet('sex_p', np.ones(2), shape=2)
        sex = pm.Categorical('sex', p=sex_p, observed=sex_transformed, shape=1)

        eastwest_p = pm.Dirichlet('eastwest_p', np.ones(2), shape=2)
        eastwest = pm.Categorical('eastwest', p=eastwest_p, observed=eastwest_transformed, shape=1)

        # priors
        inc_mu_base = pm.Uniform('inc_mu_base', 1, 1000)
        inc_mu_age = pm.Uniform('inc_mu_age', 1, 1000)
        inc_mu_educ = pm.Uniform('inc_mu_educ', 1, 2000)
        inc_mu_sex = pm.Uniform('inc_mu_sex', 1, 1000)
        inc_mu_male_west = pm.Uniform('inc_mu_male_west', 1, 1000)
        inc_mu_eastwest = pm.Uniform('inc_mu_eastwest', 1, 1000)
        inc_mu = (
            inc_mu_base +
            inc_mu_age*(age-age_min)/age_diff +
            inc_mu_educ*educ/educ_diff +
            inc_mu_sex*sex +
            inc_mu_eastwest*eastwest +
            inc_mu_male_west*sex*eastwest
            )
        inc_sigma_base = pm.Uniform('inc_sigma_base', 100, 1000)
        inc_sigma_age = pm.Uniform('inc_sigma_age', 1, 1000)
        #inc_sigma_age_max = pm.Uniform('inc_sigma_age_max', 10,90)
        inc_sigma_educ = pm.Uniform('inc_sigma_educ', 1, 1000)
        inc_sigma_sex = pm.Uniform('inc_sigma_sex', 1, 2000)
        inc_sigma_eastwest = pm.Uniform('inc_sigma_eastwest', 1, 2000)
        inc_sigma = (
            inc_sigma_base +
            inc_sigma_age*(age-age_min)/age_diff +
            inc_sigma_educ*educ/educ_diff +
            inc_sigma_sex*sex +
            inc_sigma_eastwest*eastwest
            )

        # likelihood
        income = pm.Gamma('income', mu=inc_mu, sigma=inc_sigma, observed=data['income'])

        # priors happiness
        hap_mu_base = pm.Uniform('hap_mu_base', 0, 10)
        hap_mu_income = pm.Uniform('hap_mu_income', 0, 20)
        hap_mu_sex = pm.Uniform('hap_mu_sex', -10, 10)
        hap_mu_eastwest = pm.Uniform('hap_mu_eastwest', 0, 10)
        hap_mu = (
            hap_mu_base +
            hap_mu_income*(income/inc_max) +
            hap_mu_sex*sex +
            hap_mu_eastwest*eastwest
            )
        hap_sigma_base = pm.Uniform('hap_sigma_base', 0, 5)
        hap_sigma_income = pm.Uniform('hap_sigma_income', -3, 5)
        hap_sigma_sex = pm.Uniform('hap_sigma_sex', -1, 5)
        hap_sigma_eastwest = pm.Uniform('hap_sigma_eastwest', 0, 5)
        hap_sigma = (
            hap_sigma_base +
            hap_sigma_income*(income/inc_max) +
            hap_mu_sex*sex +
            hap_sigma_eastwest*eastwest
            )

        # likelihood happiness
        happiness = pm.TruncatedNormal('happiness', mu=hap_mu, sigma=hap_sigma, lower=0, upper=10, observed=data['happiness'])

    m = ProbabilisticPymc3Model(modelname, allbus_model, shared_vars={}, data_mapping=dtm)

    if fit:
        m.fit(data)
        cll(m, test_data, model_file)
    return data, m


######################################
# Call all model generating functions
######################################
if __name__ == '__main__':
    modeldir = 'models'  # sub directory where to store created models
    mypath = "../../../fitted_models/"
    if not os.path.exists(mypath):
        os.makedirs(mypath)
    start = timeit.default_timer()
    testcasemodel_path = mypath
    testcasedata_path = mypath

    from inspect import getmembers, isfunction
    #import scripts.julien.pymc_models_bank2 as bank
    import scripts.julien.pymc_models_allbus2 as allbus

    module = allbus

    functions = [o[1] for o in getmembers(module) if isfunction(o[1]) and o[0].startswith("create")]

    model_file = "allbus.dat"
    with open(model_file, "a+") as f:
        f.write(f"modelname, happiness_acc, happiness_hit, sex_acc, sex_hit, q1, q2, q3, q4\n")

    #print(functions)

    # This list specifies which models are created when the script is run. If you only want to create
    # specific models, adjust the list accordingly

    create_functions = functions
    #create_functions = [create_hand_tuned_model]

    for func in create_functions:
        ready = False
        tried = 0
        while not ready:
            #data, m_fitted = func(fit=True)
            try:
                data, m_fitted = func(fit=True)
                ready = True
            except Exception as e:
                tried += 1
                print(f"Tried {tried} times.")


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
