#!usr/bin/python
# -*- coding: utf-8 -*-import string

import pymc3 as pm
import theano.tensor as tt

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper
from mb_modelbase.utils.Metrics import *

import scripts.experiments.allbus as allbus_data

# LOAD FILES
test_data = allbus_data.train(numeric_happy=False)
train_data = allbus_data.test(numeric_happy=False)

df = train_data

# SAVE PARAMETER IN THIS FILE
model_file = 'allbus_results.dat'
continues_data_file = 'allbus_happiness_values.dat'

sample_size = 30000

allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                       'lived_abroad': {'No': 0, 'Yes': 1}}

allbus_backward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                      'lived_abroad': {0: 'No', 1: 'Yes'}}

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)

#####################
# 48 parameter
#####################
def create_allbus_tabubiccg(filename="", modelname="allbus_tabubiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347,0.653])
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(eastwest, 0), [0.8847,0.1153], [0.7859,0.2141]))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), educ*0.2477+6.5963, educ*0.2059+7.295), sigma=tt.switch(tt.eq(eastwest, 0), 1.7444, 1.7047))
        income = pm.Normal('income', mu=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*154.9222+happiness*75.795+144.8802, educ*326.6378+happiness*116.0607+-279.4209), tt.switch(tt.eq(sex, 0), educ*220.2771+happiness*-0.0931+594.0865, educ*384.5272+happiness*184.258+-380.217)), sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 636.2516, 956.2709), tt.switch(tt.eq(sex, 0), 845.3302, 1437.4018)))
        age = pm.Normal('age', mu=tt.switch(tt.eq(eastwest, 0), educ*-4.7345+income*0.0+70.8893, educ*-5.3423+income*0.0025+65.1793), sigma=tt.switch(tt.eq(eastwest, 0), 16.4303, 16.2479))
        health = pm.Normal('health', mu=age*-0.0161+educ*0.0921+income*0.0001+happiness*0.214+2.3658, sigma=0.8404)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 94 parameter
#####################
def create_allbus_tabuaiccg(filename="", modelname="allbus_tabuaiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347,0.653])
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), 7.4376, 8.016), sigma=tt.switch(tt.eq(eastwest, 0), 1.7636, 1.7225))
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(eastwest, 0), [0.8847,0.1153], [0.7859,0.2141]))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), happiness*73.3999+654.7102, happiness*128.9098+688.5065), tt.switch(tt.eq(sex, 0), happiness*29.6378+1089.2063, happiness*182.4324+876.6178)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), happiness*209.2241+-36.0013, happiness*383.8033+-944.205), tt.switch(tt.eq(sex, 0), happiness*13.6006+1451.281, happiness*308.0543+252.2573))), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 588.4582, 960.978), tt.switch(tt.eq(sex, 0), 828.0147, 1415.2095)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 995.7663, 1325.9757), tt.switch(tt.eq(sex, 0), 1066.7325, 1776.4819))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(eastwest, 0), income*-0.0018+57.3971, income*0.0012+48.9231), sigma=tt.switch(tt.eq(eastwest, 0), 17.1271, 17.4177))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), age*-0.0235+income*0.0004+happiness*0.0467+3.765, age*-0.0093+income*0.0004+happiness*0.0363+2.8585), tt.switch(tt.eq(sex, 0), age*-0.0275+income*0.0004+happiness*0.1405+3.07, age*-0.021+income*0.0003+happiness*0.0171+3.6286)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), age*-0.0458+income*0.0+happiness*0.0842+5.7225, age*0.0055+income*0.0+happiness*0.2461+1.9035), tt.switch(tt.eq(sex, 0), age*-0.0164+income*0.0002+happiness*0.0332+4.4459, age*-0.0195+income*0.0002+happiness*-0.0266+4.4359))), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.9154, 0.9498), tt.switch(tt.eq(sex, 0), 0.9832, 1.0887)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.8719, 1.0932), tt.switch(tt.eq(sex, 0), 1.0048, 1.1611))))
        health = pm.Normal('health', mu=tt.switch(tt.eq(sex, 0), age*-0.0153+educ*0.0675+income*0.0001+happiness*0.2544+2.0699, age*-0.0173+educ*0.1062+income*0.0001+happiness*0.1757+2.6659), sigma=tt.switch(tt.eq(sex, 0), 0.8483, 0.8291))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 48 parameter
#####################
def create_allbus_hcbiccg(filename="", modelname="allbus_hcbiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347,0.653])
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(eastwest, 0), [0.8847,0.1153], [0.7859,0.2141]))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), educ*0.2477+6.5963, educ*0.2059+7.295), sigma=tt.switch(tt.eq(eastwest, 0), 1.7444, 1.7047))
        income = pm.Normal('income', mu=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*154.9222+happiness*75.795+144.8802, educ*326.6378+happiness*116.0607+-279.4209), tt.switch(tt.eq(sex, 0), educ*220.2771+happiness*-0.0931+594.0865, educ*384.5272+happiness*184.258+-380.217)), sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 636.2516, 956.2709), tt.switch(tt.eq(sex, 0), 845.3302, 1437.4018)))
        age = pm.Normal('age', mu=tt.switch(tt.eq(eastwest, 0), educ*-4.7345+income*0.0+70.8893, educ*-5.3423+income*0.0025+65.1793), sigma=tt.switch(tt.eq(eastwest, 0), 16.4303, 16.2479))
        health = pm.Normal('health', mu=age*-0.0161+educ*0.0921+income*0.0001+happiness*0.214+2.3658, sigma=0.8404)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 104 parameter
#####################
def create_allbus_hcaiccg(filename="", modelname="allbus_hcaiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347,0.653])
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), 7.4376, 8.016), sigma=tt.switch(tt.eq(eastwest, 0), 1.7636, 1.7225))
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(eastwest, 0), [0.8847,0.1153], [0.7859,0.2141]))
        age = pm.Normal('age', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), 55.2714, 51.7959), tt.switch(tt.eq(eastwest, 0), 51.0548, 48.9451)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), 17.0513, 17.4567), tt.switch(tt.eq(eastwest, 0), 17.9086, 17.4475)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), age*-0.0234+happiness*0.0788+4.0463, age*-0.0119+happiness*0.0887+3.2824), tt.switch(tt.eq(sex, 0), age*-0.0275+happiness*0.1534+3.5409, age*-0.018+happiness*0.0702+3.7171)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), age*-0.0463+happiness*0.094+5.7346, age*0.0056+happiness*0.2405+1.9092), tt.switch(tt.eq(sex, 0), age*-0.016+happiness*0.0352+4.6898, age*-0.0164+happiness*0.0499+4.3364))), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.9494, 1.0251), tt.switch(tt.eq(sex, 0), 1.0461, 1.1607)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.8587, 1.0781), tt.switch(tt.eq(sex, 0), 1.0179, 1.2366))))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), age*4.1501+educ*168.7726+happiness*60.1355+-40.0377, age*-2.1667+educ*354.6403+happiness*97.0978+-123.8215), tt.switch(tt.eq(sex, 0), age*7.6193+educ*272.9141+happiness*-12.2441+116.8889, age*18.2228+educ*422.785+happiness*154.2527+-1265.2314)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), age*-8.243+educ*55.0251+happiness*231.3704+-22.6871, age*-7.5497+educ*-24.1353+happiness*366.4571+-319.055), tt.switch(tt.eq(sex, 0), age*4.9413+educ*194.6029+happiness*4.4131+474.9395, age*20.6762+educ*505.5709+happiness*284.8966+-2595.817))), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 568.4922, 886.0709), tt.switch(tt.eq(sex, 0), 779.0356, 1317.2336)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 1006.503, 1356.3559), tt.switch(tt.eq(sex, 0), 1057.1919, 1662.058))))
        health = pm.Normal('health', mu=tt.switch(tt.eq(sex, 0), age*-0.0153+educ*0.0675+income*0.0001+happiness*0.2544+2.0699, age*-0.0173+educ*0.1062+income*0.0001+happiness*0.1757+2.6659), sigma=tt.switch(tt.eq(sex, 0), 0.8483, 0.8291))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 56 parameter
#####################
def create_allbus_iamb(filename="", modelname="allbus_iamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8202,0.1798])
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        eastwest = pm.Categorical('eastwest', p=tt.switch(tt.eq(lived_abroad, 0), [0.3743,0.6257], [0.2226,0.7774]))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), 7.4376, 8.016), sigma=tt.switch(tt.eq(eastwest, 0), 1.7636, 1.7225))
        age = pm.Normal('age', mu=educ*-4.6886+68.6804, sigma=16.5729)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*142.1342+happiness*61.5163+270.2846, educ*361.5785+happiness*96.5958+-261.5227), tt.switch(tt.eq(sex, 0), educ*225.0456+happiness*-3.6509+597.9953, educ*352.8756+happiness*156.7481+-94.7012)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*145.0125+happiness*212.5877+-662.9993, educ*-34.9981+happiness*391.629+-862.6152), tt.switch(tt.eq(sex, 0), educ*173.5222+happiness*11.0788+734.7469, educ*441.8308+happiness*284.7614+-1278.1998))), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 571.1388, 885.3193), tt.switch(tt.eq(sex, 0), 787.4688, 1351.6334)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 996.0851, 1343.732), tt.switch(tt.eq(sex, 0), 1055.6976, 1692.5275))))
        health = pm.Normal('health', mu=age*-0.0156+educ*0.1089+happiness*0.2189+2.3369, sigma=0.8424)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 56 parameter
#####################
def create_allbus_fastiamb(filename="", modelname="allbus_fastiamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8202,0.1798])
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        eastwest = pm.Categorical('eastwest', p=tt.switch(tt.eq(lived_abroad, 0), [0.3743,0.6257], [0.2226,0.7774]))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), 7.4376, 8.016), sigma=tt.switch(tt.eq(eastwest, 0), 1.7636, 1.7225))
        age = pm.Normal('age', mu=educ*-4.6886+68.6804, sigma=16.5729)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*142.1342+happiness*61.5163+270.2846, educ*361.5785+happiness*96.5958+-261.5227), tt.switch(tt.eq(sex, 0), educ*225.0456+happiness*-3.6509+597.9953, educ*352.8756+happiness*156.7481+-94.7012)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*145.0125+happiness*212.5877+-662.9993, educ*-34.9981+happiness*391.629+-862.6152), tt.switch(tt.eq(sex, 0), educ*173.5222+happiness*11.0788+734.7469, educ*441.8308+happiness*284.7614+-1278.1998))), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 571.1388, 885.3193), tt.switch(tt.eq(sex, 0), 787.4688, 1351.6334)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 996.0851, 1343.732), tt.switch(tt.eq(sex, 0), 1055.6976, 1692.5275))))
        health = pm.Normal('health', mu=age*-0.0156+educ*0.1089+happiness*0.2189+2.3369, sigma=0.8424)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, continues_data_file)
    return df, m

#####################
# 56 parameter
#####################
def create_allbus_interiamb(filename="", modelname="allbus_interiamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347,0.653])
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(eastwest, 0), 7.4376, 8.016), sigma=tt.switch(tt.eq(eastwest, 0), 1.7636, 1.7225))
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(eastwest, 0), [0.8847,0.1153], [0.7859,0.2141]))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3342, 4.061), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1388, 1.1689))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*142.1342+happiness*61.5163+270.2846, educ*361.5785+happiness*96.5958+-261.5227), tt.switch(tt.eq(sex, 0), educ*225.0456+happiness*-3.6509+597.9953, educ*352.8756+happiness*156.7481+-94.7012)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), educ*145.0125+happiness*212.5877+-662.9993, educ*-34.9981+happiness*391.629+-862.6152), tt.switch(tt.eq(sex, 0), educ*173.5222+happiness*11.0788+734.7469, educ*441.8308+happiness*284.7614+-1278.1998))), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 571.1388, 885.3193), tt.switch(tt.eq(sex, 0), 787.4688, 1351.6334)), tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 996.0851, 1343.732), tt.switch(tt.eq(sex, 0), 1055.6976, 1692.5275))))
        age = pm.Normal('age', mu=educ*-4.6886+68.6804, sigma=16.5729)
        health = pm.Normal('health', mu=age*-0.0156+educ*0.1089+happiness*0.2189+2.3369, sigma=0.8424)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file, continues_data_file)
    return df, m

