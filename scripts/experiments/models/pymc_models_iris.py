#!usr/bin/python
# -*- coding: utf-8 -*-import string

import pymc3 as pm
import theano.tensor as tt

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper
from mb_modelbase.utils.Metrics import *

import scripts.experiments.iris as iris_data

# LOAD FILES
df = iris_data.iris(discrete_species=False)
train_data = df
test_data = df

# SAVE PARAMETER IN THIS FILE
model_file = 'allbus_results.dat'

sample_size = 3000

continues_data_file = None

iris_forward_map = {'species': {'setosa': 0, 'versicolor': 1, 'virginica': 2}}

iris_backward_map = {'species': {0: 'setosa', 1: 'versicolor', 2: 'virginica'}}

dtm = DataTypeMapper()
for name, map_ in iris_backward_map.items():
    dtm.set_map(forward=iris_forward_map[name], backward=map_, name=name)

#####################
# 32 parameter
#####################
def create_iris_tabubiccg(filename="", modelname="iris_tabubiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), 0.246, tt.switch(tt.eq(species, 1), 1.326, 2.026)), sigma=tt.switch(tt.eq(species, 0), 0.1054, tt.switch(tt.eq(species, 1), 0.1978, 0.2747)))
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), petalwidth*0.8372+3.2221, tt.switch(tt.eq(species, 1), petalwidth*1.0536+1.3729, petalwidth*0.6314+1.6948)), sigma=tt.switch(tt.eq(species, 0), 0.3725, tt.switch(tt.eq(species, 1), 0.2371, 0.2747)))
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), petalwidth*0.5465+1.3276, tt.switch(tt.eq(species, 1), petalwidth*1.8693+1.7813, petalwidth*0.6473+4.2407)), sigma=tt.switch(tt.eq(species, 0), 0.1655, tt.switch(tt.eq(species, 1), 0.2931, 0.5279)))
        sepallength = pm.Normal('sepallength', mu=sepalwidth*0.6508+petallength*0.7091+petalwidth*-0.5565+1.856, sigma=0.3145)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 39 parameter
#####################
def create_iris_tabuaiccg(filename="", modelname="iris_tabuaiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), 0.246, tt.switch(tt.eq(species, 1), 1.326, 2.026)), sigma=tt.switch(tt.eq(species, 0), 0.1054, tt.switch(tt.eq(species, 1), 0.1978, 0.2747)))
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), petalwidth*0.5465+1.3276, tt.switch(tt.eq(species, 1), petalwidth*1.8693+1.7813, petalwidth*0.6473+4.2407)), sigma=tt.switch(tt.eq(species, 0), 0.1655, tt.switch(tt.eq(species, 1), 0.2931, 0.5279)))
        sepallength = pm.Normal('sepallength', mu=tt.switch(tt.eq(species, 0), petallength*0.5423+4.2132, tt.switch(tt.eq(species, 1), petallength*0.8283+2.4075, petallength*0.9957+1.0597)), sigma=tt.switch(tt.eq(species, 0), 0.3432, tt.switch(tt.eq(species, 1), 0.3425, 0.3232)))
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), sepallength*0.79+petalwidth*0.1023+-0.552, tt.switch(tt.eq(species, 1), sepallength*0.1413+petalwidth*0.8521+0.8012, sepallength*0.1685+petalwidth*0.5217+0.8066)), sigma=tt.switch(tt.eq(species, 0), 0.259, tt.switch(tt.eq(species, 1), 0.2313, 0.257)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 32 parameter
#####################
def create_iris_hcbiccg(filename="", modelname="iris_hcbiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), 1.462, tt.switch(tt.eq(species, 1), 4.26, 5.552)), sigma=tt.switch(tt.eq(species, 0), 0.1737, tt.switch(tt.eq(species, 1), 0.4699, 0.5519)))
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), petallength*0.2012+-0.0482, tt.switch(tt.eq(species, 1), petallength*0.3311+-0.0843, petallength*0.1603+1.136)), sigma=tt.switch(tt.eq(species, 0), 0.1005, tt.switch(tt.eq(species, 1), 0.1234, 0.2627)))
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), petalwidth*0.8372+3.2221, tt.switch(tt.eq(species, 1), petalwidth*1.0536+1.3729, petalwidth*0.6314+1.6948)), sigma=tt.switch(tt.eq(species, 0), 0.3725, tt.switch(tt.eq(species, 1), 0.2371, 0.2747)))
        sepallength = pm.Normal('sepallength', mu=sepalwidth*0.6508+petallength*0.7091+petalwidth*-0.5565+1.856, sigma=0.3145)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 39 parameter
#####################
def create_iris_hcaiccg(filename="", modelname="iris_hcaiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), 1.462, tt.switch(tt.eq(species, 1), 4.26, 5.552)), sigma=tt.switch(tt.eq(species, 0), 0.1737, tt.switch(tt.eq(species, 1), 0.4699, 0.5519)))
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), petallength*0.2012+-0.0482, tt.switch(tt.eq(species, 1), petallength*0.3311+-0.0843, petallength*0.1603+1.136)), sigma=tt.switch(tt.eq(species, 0), 0.1005, tt.switch(tt.eq(species, 1), 0.1234, 0.2627)))
        sepallength = pm.Normal('sepallength', mu=tt.switch(tt.eq(species, 0), petallength*0.5423+4.2132, tt.switch(tt.eq(species, 1), petallength*0.8283+2.4075, petallength*0.9957+1.0597)), sigma=tt.switch(tt.eq(species, 0), 0.3432, tt.switch(tt.eq(species, 1), 0.3425, 0.3232)))
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), sepallength*0.79+petalwidth*0.1023+-0.552, tt.switch(tt.eq(species, 1), sepallength*0.1413+petalwidth*0.8521+0.8012, sepallength*0.1685+petalwidth*0.5217+0.8066)), sigma=tt.switch(tt.eq(species, 0), 0.259, tt.switch(tt.eq(species, 1), 0.2313, 0.257)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 36 parameter
#####################
def create_iris_gs(filename="", modelname="iris_gs", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        sepallength = pm.Normal('sepallength', mu=tt.switch(tt.eq(species, 0), 5.006, tt.switch(tt.eq(species, 1), 5.936, 6.588)), sigma=tt.switch(tt.eq(species, 0), 0.3525, tt.switch(tt.eq(species, 1), 0.5162, 0.6359)))
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), 0.246, tt.switch(tt.eq(species, 1), 1.326, 2.026)), sigma=tt.switch(tt.eq(species, 0), 0.1054, tt.switch(tt.eq(species, 1), 0.1978, 0.2747)))
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), petalwidth*0.8372+3.2221, tt.switch(tt.eq(species, 1), petalwidth*1.0536+1.3729, petalwidth*0.6314+1.6948)), sigma=tt.switch(tt.eq(species, 0), 0.3725, tt.switch(tt.eq(species, 1), 0.2371, 0.2747)))
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), sepallength*0.0934+petalwidth*0.4596+0.8813, tt.switch(tt.eq(species, 1), sepallength*0.4208+petalwidth*1.2692+0.0795, sepallength*0.7291+petalwidth*0.1727+0.3987)), sigma=tt.switch(tt.eq(species, 0), 0.1641, tt.switch(tt.eq(species, 1), 0.2308, 0.2797)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 36 parameter
#####################
def create_iris_iamb(filename="", modelname="iris_iamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), 1.462, tt.switch(tt.eq(species, 1), 4.26, 5.552)), sigma=tt.switch(tt.eq(species, 0), 0.1737, tt.switch(tt.eq(species, 1), 0.4699, 0.5519)))
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), petallength*0.2012+-0.0482, tt.switch(tt.eq(species, 1), petallength*0.3311+-0.0843, petallength*0.1603+1.136)), sigma=tt.switch(tt.eq(species, 0), 0.1005, tt.switch(tt.eq(species, 1), 0.1234, 0.2627)))
        sepallength = pm.Normal('sepallength', mu=tt.switch(tt.eq(species, 0), petallength*0.5423+4.2132, tt.switch(tt.eq(species, 1), petallength*0.8283+2.4075, petallength*0.9957+1.0597)), sigma=tt.switch(tt.eq(species, 0), 0.3432, tt.switch(tt.eq(species, 1), 0.3425, 0.3232)))
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), petalwidth*0.8372+3.2221, tt.switch(tt.eq(species, 1), petalwidth*1.0536+1.3729, petalwidth*0.6314+1.6948)), sigma=tt.switch(tt.eq(species, 0), 0.3725, tt.switch(tt.eq(species, 1), 0.2371, 0.2747)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 36 parameter
#####################
def create_iris_fastiamb(filename="", modelname="iris_fastiamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), 1.462, tt.switch(tt.eq(species, 1), 4.26, 5.552)), sigma=tt.switch(tt.eq(species, 0), 0.1737, tt.switch(tt.eq(species, 1), 0.4699, 0.5519)))
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), petallength*0.2012+-0.0482, tt.switch(tt.eq(species, 1), petallength*0.3311+-0.0843, petallength*0.1603+1.136)), sigma=tt.switch(tt.eq(species, 0), 0.1005, tt.switch(tt.eq(species, 1), 0.1234, 0.2627)))
        sepallength = pm.Normal('sepallength', mu=tt.switch(tt.eq(species, 0), petallength*0.5423+4.2132, tt.switch(tt.eq(species, 1), petallength*0.8283+2.4075, petallength*0.9957+1.0597)), sigma=tt.switch(tt.eq(species, 0), 0.3432, tt.switch(tt.eq(species, 1), 0.3425, 0.3232)))
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), petalwidth*0.8372+3.2221, tt.switch(tt.eq(species, 1), petalwidth*1.0536+1.3729, petalwidth*0.6314+1.6948)), sigma=tt.switch(tt.eq(species, 0), 0.3725, tt.switch(tt.eq(species, 1), 0.2371, 0.2747)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 36 parameter
#####################
def create_iris_interiamb(filename="", modelname="iris_interiamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        species = pm.Categorical('species', p=[0.3333,0.3333,0.3333])
        sepalwidth = pm.Normal('sepalwidth', mu=tt.switch(tt.eq(species, 0), 3.428, tt.switch(tt.eq(species, 1), 2.77, 2.974)), sigma=tt.switch(tt.eq(species, 0), 0.3791, tt.switch(tt.eq(species, 1), 0.3138, 0.3225)))
        petalwidth = pm.Normal('petalwidth', mu=tt.switch(tt.eq(species, 0), sepalwidth*0.0647+0.0242, tt.switch(tt.eq(species, 1), sepalwidth*0.4184+0.1669, sepalwidth*0.4579+0.6641)), sigma=tt.switch(tt.eq(species, 0), 0.1036, tt.switch(tt.eq(species, 1), 0.1494, 0.234)))
        petallength = pm.Normal('petallength', mu=tt.switch(tt.eq(species, 0), petalwidth*0.5465+1.3276, tt.switch(tt.eq(species, 1), petalwidth*1.8693+1.7813, petalwidth*0.6473+4.2407)), sigma=tt.switch(tt.eq(species, 0), 0.1655, tt.switch(tt.eq(species, 1), 0.2931, 0.5279)))
        sepallength = pm.Normal('sepallength', mu=tt.switch(tt.eq(species, 0), petallength*0.5423+4.2132, tt.switch(tt.eq(species, 1), petallength*0.8283+2.4075, petallength*0.9957+1.0597)), sigma=tt.switch(tt.eq(species, 0), 0.3432, tt.switch(tt.eq(species, 1), 0.3425, 0.3232)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_iris(m, test_data, model_file, continues_data_file)
    return df, m

