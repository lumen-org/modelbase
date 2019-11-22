#!usr/bin/python
# -*- coding: utf-8 -*-import string

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import theano

import math
import os

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model

import pandas as pd

filepath = os.path.join(os.path.dirname(__file__), "titanic_cleaned.csv")
df = pd.read_csv(filepath)

sample_size = 100000


###############################################################
# 158 parameter
# continuous_variables = ['age', 'fare', 'ticket']
###############################################################
def create_titanic_model_1(filename="", modelname="titanic_model_1", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    import pymc3 as pm
    titanic_model = pm.Model()
    data = None
    with titanic_model:
        survived = pm.Categorical('survived', p=[0.0185, 0.9815])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(survived, 0), [0.1111, 0.8889], [0.6667, 0.3333]))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        pclass = pm.Categorical('pclass',
                                p=tt.switch(tt.eq(sex, 0), [0.4326, 0.2696, 0.2978], [0.3772, 0.1557, 0.4671]))
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))

    m = ProbabilisticPymc3Model(modelname, titanic_model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# continuous_variables = ["ticket"]
# 376 parameter
#####################
def create_titanic_model_2(filename="", modelname="titanic_model_2", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    titanic_model = pm.Model()
    data = None
    with titanic_model:
        survived = pm.Categorical('survived', p=[0.0185, 0.9815])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(survived, 0), [0.1111, 0.8889], [0.6667, 0.3333]))
        age = pm.Categorical('age',
                             p=[0.0021, 0.0021, 0.0021, 0.0041, 0.0062, 0.0041, 0.0144, 0.0082, 0.0103, 0.0144, 0.0082,
                                0.0062, 0.0041, 0.0082, 0.0082, 0.0021, 0.0062, 0.0062, 0.0062, 0.0062, 0.0165, 0.0123,
                                0.0267, 0.0226, 0.0165, 0.0226, 0.0412, 0.0185, 0.0453, 0.0247, 0.0206, 0.0267, 0.0144,
                                0.0267, 0.142, 0.0309, 0.0226, 0.0247, 0.0021, 0.0165, 0.0123, 0.0267, 0.0329, 0.0021,
                                0.0041, 0.0123, 0.0165, 0.0123, 0.0041, 0.0082, 0.0062, 0.0062, 0.0288, 0.0041, 0.0206,
                                0.0103, 0.0123, 0.0062, 0.0062, 0.0082, 0.0103, 0.0082, 0.0041, 0.0062, 0.0021, 0.0082,
                                0.0041, 0.0041, 0.0041, 0.0021, 0.0021])
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        fare = pm.Categorical('fare',
                              p=[0.0041, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0103, 0.0123, 0.0062, 0.0021, 0.0041,
                                 0.0021, 0.0062, 0.0021, 0.0021, 0.0062, 0.0021, 0.0309, 0.0144, 0.0021, 0.0103, 0.0021,
                                 0.0021, 0.0082, 0.0062, 0.0021, 0.0021, 0.0144, 0.0165, 0.0021, 0.0041, 0.0021, 0.0021,
                                 0.0021, 0.0041, 0.0062, 0.0021, 0.0021, 0.0226, 0.0062, 0.0041, 0.0021, 0.0041, 0.0041,
                                 0.0082, 0.0021, 0.0329, 0.0041, 0.0041, 0.0021, 0.0062, 0.0041, 0.0021, 0.0021, 0.0021,
                                 0.0041, 0.0123, 0.0062, 0.0041, 0.0062, 0.0041, 0.0041, 0.0062, 0.0021, 0.0062, 0.0062,
                                 0.0041, 0.0062, 0.0021, 0.0082, 0.0041, 0.0021, 0.0041, 0.0062, 0.0103, 0.0062, 0.0062,
                                 0.0103, 0.0062, 0.0021, 0.0021, 0.0021, 0.0021, 0.0041, 0.037, 0.0082, 0.0021, 0.0062,
                                 0.0021, 0.0206, 0.0021, 0.0021, 0.0123, 0.0062, 0.0021, 0.0021, 0.0062, 0.0041, 0.0123,
                                 0.0082, 0.0021, 0.0041, 0.0062, 0.0021, 0.0041, 0.0041, 0.0062, 0.0062, 0.0041, 0.0123,
                                 0.0041, 0.0041, 0.0062, 0.0021, 0.0021, 0.0041, 0.0021, 0.0082, 0.0082, 0.0082, 0.0041,
                                 0.0082, 0.0021, 0.0123, 0.0041, 0.0041, 0.0041, 0.0041, 0.0062, 0.0021, 0.0021, 0.0021,
                                 0.0021, 0.0041, 0.0062, 0.0021, 0.0041, 0.0021, 0.0021, 0.0021, 0.0021, 0.0041, 0.0062,
                                 0.0062, 0.0041, 0.0041, 0.0062, 0.0041, 0.0041, 0.0062, 0.0021, 0.0041, 0.0103, 0.0021,
                                 0.0062, 0.0041, 0.0082, 0.0041, 0.0041, 0.0041, 0.0041, 0.0062, 0.0041, 0.0082, 0.0041,
                                 0.0103, 0.0062, 0.0021, 0.0021, 0.0062, 0.0041, 0.0062, 0.0082, 0.0041, 0.0021, 0.0062,
                                 0.0041, 0.0123, 0.0082, 0.0082])
        pclass = pm.Categorical('pclass',
                                p=tt.switch(tt.eq(sex, 0), [0.4326, 0.2696, 0.2978], [0.3772, 0.1557, 0.4671]))
        ticket = pm.Normal('ticket',
                           mu=tt.switch(tt.eq(pclass, 0), 128.8706, tt.switch(tt.eq(pclass, 1), 175.9464, 191.1503)),
                           sigma=tt.switch(tt.eq(pclass, 0), 115.7033, tt.switch(tt.eq(pclass, 1), 86.3896, 61.4168)))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))

        #data = pm.trace_to_dataframe(pm.sample(10000))
        #data.sort_index(inplace=True)
    m = ProbabilisticPymc3Model(modelname, titanic_model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

###############################################################
# continuous_variables = ['age', 'fare', 'ticket']
# 158 parameter
###############################################################
def create_titanic_model_3(filename="", modelname="titanic_model_3", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    # define model explicitely (no fitting to data with pymc3)
    titanic_model = pm.Model()
    with titanic_model:
        survived = pm.Categorical('survived', p=[0.0185, 0.9815])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(survived, 0), [0.1111, 0.8889], [0.6667, 0.3333]))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        pclass = pm.Categorical('pclass',
                                p=tt.switch(tt.eq(sex, 0), [0.4326, 0.2696, 0.2978], [0.3772, 0.1557, 0.4671]))
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))

    m = ProbabilisticPymc3Model(modelname, titanic_model)
    m.nr_of_posterior_samples = sample_size
    # import pandas as pd

    # filepath = os.path.join(os.path.dirname(__file__), "titanic_cleaned.csv")
    # df = pd.read_csv(filepath)
    if fit:
        m.fit(df, auto_extend=False)
    return df, m



###############################################################
# continuous_variables = ['age', 'fare', 'ticket']
# whitelist = [("pclass", "survived"), ("sex", "survived")]
# 159 parameter
###############################################################
def create_titanic_model_4(filename="", modelname="titanic_model_4", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    titanic_model = pm.Model()
    data = None
    with titanic_model:
        pclass = pm.Categorical('pclass', p=[0.4136, 0.2305, 0.356])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(pclass, 0), [0.6866, 0.3134],
                                                tt.switch(tt.eq(pclass, 1), [0.7679, 0.2321], [0.5491, 0.4509])))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        survived = pm.Categorical('survived', p=tt.switch(tt.eq(sex, 0), [0.0031, 0.9969], [0.0479, 0.9521]))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))

    m = ProbabilisticPymc3Model(modelname, titanic_model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m


###############################################################
# continuous_variables = ['age', 'fare', 'ticket']
# whitelist = [("pclass", "survived"), ("sex", "survived")]
# 167 parameter
###############################################################
def create_titanic_model_5(filename="", modelname="titanic_model_5", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    titanic_model = pm.Model()
    data = None
    with titanic_model:
        pclass = pm.Categorical('pclass', p=[0.4136, 0.2305, 0.356])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(pclass, 0), [0.6866, 0.3134],
                                                tt.switch(tt.eq(pclass, 1), [0.7679, 0.2321], [0.5491, 0.4509])))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        survived = pm.Categorical('survived',
                                  p=tt.switch(tt.eq(pclass, 0), tt.switch(tt.eq(sex, 0), [0.0, 1.0], [0.0317, 0.9683]),
                                              tt.switch(tt.eq(pclass, 1),
                                                        tt.switch(tt.eq(sex, 0), [0.0, 1.0], [0.0385, 0.9615]),
                                                        tt.switch(tt.eq(sex, 0), [0.0105, 0.9895], [0.0641, 0.9359]))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))


    data_map = {}

    raise NotImplementedError("Data Map")
    m = ProbabilisticPymc3Model(modelname, titanic_model, data_map)

    """
    nodes = ['age', 'sex', ...]
    edges = [('age', 'sex'), ...]
    blacklist
    whitelist
    continuous
    """

    m.set_gm_graph()



    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

