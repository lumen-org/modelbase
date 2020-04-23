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
from mb_modelbase.utils.data_type_mapper import DataTypeMapper

import pandas as pd

filepath = os.path.join(os.path.dirname(__file__), "data", "allbus.csv")
df = pd.read_csv(filepath)

sample_size = 100000

allbus_backward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                       'lived_abroad': {'No': 0, 'Yes': 1}
                       }

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward='auto', backward=map_, name=name)

#####################
# 114 parameter
#####################
def create_allbus_tabu_loglikcg(filename="", modelname="allbus_tabu_loglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(lived_abroad, 0), [0.4863,0.5137], [0.4539,0.5461]))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.6445, 0.6174), tt.switch(tt.eq(sex, 0), 0.8021, 0.7792)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.4789, 0.4863), tt.switch(tt.eq(sex, 0), 0.3995, 0.4157)))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*0.4998+7.5483, eastwest*0.5703+7.3397), tt.switch(tt.eq(sex, 0), eastwest*0.6012+7.7105, eastwest*0.3559+7.6275)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.7528, 1.7169), tt.switch(tt.eq(sex, 0), 1.5438, 1.6729)))
        health = pm.Normal('health', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*0.1351+happiness*0.2455+1.5642, eastwest*-0.0133+happiness*0.1928+2.0811), tt.switch(tt.eq(sex, 0), eastwest*-0.0037+happiness*0.303+1.4271, eastwest*-0.1941+happiness*0.269+1.7522)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.9097, 0.8876), tt.switch(tt.eq(sex, 0), 0.8722, 0.9391)))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*52.4719+happiness*30.2713+health*90.9186+688.4361, eastwest*560.8984+happiness*133.9881+health*138.0922+161.7676), tt.switch(tt.eq(sex, 0), eastwest*-24.9051+happiness*61.542+health*-12.157+1105.1747, eastwest*646.4794+happiness*267.212+health*146.1254+-679.192)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 774.9052, 1237.3257), tt.switch(tt.eq(sex, 0), 997.8213, 1579.3942)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), income*0.0004+eastwest*-0.0857+happiness*0.0471+health*0.1932+1.8236, income*0.0002+eastwest*-0.1542+happiness*-0.0119+health*0.2758+2.0366), tt.switch(tt.eq(sex, 0), income*0.0002+eastwest*0.027+happiness*-0.086+health*0.3679+3.1361, income*0.0002+eastwest*-0.1625+happiness*-0.0401+health*0.2618+2.87)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.0447, 1.0748), tt.switch(tt.eq(sex, 0), 1.0249, 1.1456)))
        age = pm.Normal('age', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), educ*-5.8253+income*0.0032+eastwest*-3.9383+happiness*1.8691+health*-5.0885+74.4119, educ*-3.4663+income*0.0024+eastwest*-3.9209+happiness*1.0883+health*-6.8246+78.4069), tt.switch(tt.eq(sex, 0), educ*-5.4212+income*0.0009+eastwest*-3.7137+happiness*2.9171+health*-4.5003+64.6867, educ*-1.0406+income*0.0015+eastwest*-0.2426+happiness*0.9936+health*-5.8061+65.1705)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 15.0116, 15.5783), tt.switch(tt.eq(sex, 0), 15.2106, 16.5573)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 72 parameter
#####################
def create_allbus_tabu_aiccg(filename="", modelname="allbus_tabu_aiccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        age = pm.Normal('age', mu=tt.switch(tt.eq(lived_abroad, 0), 53.1309, 48.9693), sigma=tt.switch(tt.eq(lived_abroad, 0), 17.3465, 17.4244))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*-0.0245+4.6575, age*-0.0174+4.2596), tt.switch(tt.eq(sex, 0), age*-0.0249+5.3784, age*-0.0074+4.2735)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.0465, 1.1167), tt.switch(tt.eq(sex, 0), 1.0003, 1.2143)))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0026+0.7671, age*-0.0002+0.8002), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4808, 0.4085))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*6.6264+educ*258.9081+eastwest*112.1817+-6.3859, age*11.2241+educ*379.6058+eastwest*669.0161+-246.3414), tt.switch(tt.eq(sex, 0), age*5.4384+educ*212.5717+eastwest*17.0135+378.0212, age*10.8018+educ*423.9158+eastwest*731.6931+-290.5039)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 736.4412, 1198.2071), tt.switch(tt.eq(sex, 0), 978.8413, 1578.9939)))
        health = pm.Normal('health', mu=tt.switch(tt.eq(sex, 0), age*-0.0133+educ*0.1332+income*0.0001+eastwest*0.1897+3.6014, age*-0.0182+educ*0.1163+income*0.0001+eastwest*-0.0397+3.9413), sigma=tt.switch(tt.eq(sex, 0), 0.9475, 0.8755))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(sex, 0), age*0.0203+educ*0.1757+income*0.0+eastwest*0.3886+health*0.7865+3.0787, age*0.0103+educ*-0.0054+income*0.0002+eastwest*0.3731+health*0.6639+4.1456), sigma=tt.switch(tt.eq(sex, 0), 1.5127, 1.5539))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 37 parameter
#####################
def create_allbus_tabu_biccg(filename="", modelname="allbus_tabu_biccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3446, 4.0426), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1427, 1.1696))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=educ*0.2138+eastwest*0.5294+6.7339, sigma=1.6968)
        health = pm.Normal('health', mu=educ*0.181+happiness*0.2148+1.3035, sigma=0.8801)
        income = pm.Normal('income', mu=tt.switch(tt.eq(sex, 0), educ*211.6522+eastwest*64.2077+happiness*31.6485+294.2901, educ*330.5615+eastwest*586.4313+happiness*160.9824+-645.3277), sigma=tt.switch(tt.eq(sex, 0), 787.7015, 1264.8692))
        age = pm.Normal('age', mu=educ*-4.2074+income*0.0021+eastwest*-3.5511+happiness*1.53+health*-5.7348+74.4703, sigma=15.534)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 37 parameter
#####################
def create_allbus_tabu_predloglikcg(filename="", modelname="allbus_tabu_predloglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3446, 4.0426), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1427, 1.1696))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=educ*0.2138+eastwest*0.5294+6.7339, sigma=1.6968)
        health = pm.Normal('health', mu=educ*0.181+happiness*0.2148+1.3035, sigma=0.8801)
        income = pm.Normal('income', mu=tt.switch(tt.eq(sex, 0), educ*211.6522+eastwest*64.2077+happiness*31.6485+294.2901, educ*330.5615+eastwest*586.4313+happiness*160.9824+-645.3277), sigma=tt.switch(tt.eq(sex, 0), 787.7015, 1264.8692))
        age = pm.Normal('age', mu=educ*-4.2074+income*0.0021+eastwest*-3.5511+happiness*1.53+health*-5.7348+74.4703, sigma=15.534)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 114 parameter
#####################
def create_allbus_hc_loglikcg(filename="", modelname="allbus_hc_loglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(sex, 0), [0.8247,0.1753], [0.8051,0.1949]))
        age = pm.Normal('age', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 52.9369, 53.3145), tt.switch(tt.eq(sex, 0), 46.6042, 50.9351)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 17.2198, 17.4726), tt.switch(tt.eq(sex, 0), 17.1287, 17.4603)))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*-0.0032+0.8152, age*-0.002+0.7215), tt.switch(tt.eq(sex, 0), age*-0.0016+0.8778, age*0.0011+0.7257)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.476, 0.4853), tt.switch(tt.eq(sex, 0), 0.3995, 0.4162)))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*0.0012+eastwest*0.5048+7.4822, age*-0.0013+eastwest*0.5671+7.409), tt.switch(tt.eq(sex, 0), age*0.0114+eastwest*0.6352+7.1522, age*-0.0018+eastwest*0.3592+7.7156)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.7537, 1.7176), tt.switch(tt.eq(sex, 0), 1.5355, 1.6763)))
        health = pm.Normal('health', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*-0.0173+eastwest*0.0619+happiness*0.2475+2.5132, age*-0.0196+eastwest*-0.0613+happiness*0.1902+3.1754), tt.switch(tt.eq(sex, 0), age*-0.017+eastwest*-0.0689+happiness*0.3268+2.0766, age*-0.0169+eastwest*-0.1616+happiness*0.2657+2.6144)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.8605, 0.8195), tt.switch(tt.eq(sex, 0), 0.8248, 0.8931)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*-0.0234+eastwest*-0.1437+happiness*0.0975+health*0.0867+3.6135, age*-0.0132+eastwest*-0.0496+happiness*0.0388+health*0.2095+3.0245), tt.switch(tt.eq(sex, 0), age*-0.022+eastwest*-0.0631+happiness*0.0012+health*0.2225+4.4149, age*-0.0025+eastwest*-0.0319+happiness*0.0166+health*0.276+2.8903)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.0239, 1.0968), tt.switch(tt.eq(sex, 0), 0.9847, 1.1873)))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*7.698+educ*246.3672+eastwest*94.4126+happiness*3.1363+health*81.3826+-325.8667, age*13.3169+educ*339.5376+eastwest*600.2035+happiness*109.0891+health*133.7577+-1496.6301), tt.switch(tt.eq(sex, 0), age*3.8995+educ*214.9521+eastwest*-14.5387+happiness*64.0641+health*-65.358+195.4509, age*12.7955+educ*359.5552+eastwest*648.6692+happiness*244.7378+health*116.8458+-2446.7166)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 732.8992, 1172.662), tt.switch(tt.eq(sex, 0), 980.3736, 1514.6133)))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 72 parameter
#####################
def create_allbus_hc_aiccg(filename="", modelname="allbus_hc_aiccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        age = pm.Normal('age', mu=tt.switch(tt.eq(lived_abroad, 0), eastwest*-3.3178+55.223, eastwest*-0.3947+49.2809), sigma=tt.switch(tt.eq(lived_abroad, 0), 17.2771, 17.4443))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*-0.0245+4.6575, age*-0.0174+4.2596), tt.switch(tt.eq(sex, 0), age*-0.0249+5.3784, age*-0.0074+4.2735)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.0465, 1.1167), tt.switch(tt.eq(sex, 0), 1.0003, 1.2143)))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), age*6.6264+educ*258.9081+eastwest*112.1817+-6.3859, age*11.2241+educ*379.6058+eastwest*669.0161+-246.3414), tt.switch(tt.eq(sex, 0), age*5.4384+educ*212.5717+eastwest*17.0135+378.0212, age*10.8018+educ*423.9158+eastwest*731.6931+-290.5039)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 736.4412, 1198.2071), tt.switch(tt.eq(sex, 0), 978.8413, 1578.9939)))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(sex, 0), age*0.0098+educ*0.2804+income*0.0001+eastwest*0.5379+5.9111, age*-0.0018+educ*0.0718+income*0.0003+eastwest*0.3468+6.7624), sigma=tt.switch(tt.eq(sex, 0), 1.6857, 1.6585))
        health = pm.Normal('health', mu=tt.switch(tt.eq(sex, 0), age*-0.0157+educ*0.0635+income*0.0001+eastwest*0.0561+happiness*0.2485+2.1325, age*-0.0179+educ*0.103+income*0.0001+eastwest*-0.1038+happiness*0.185+2.6903), sigma=tt.switch(tt.eq(sex, 0), 0.8503, 0.8203))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 37 parameter
#####################
def create_allbus_hc_biccg(filename="", modelname="allbus_hc_biccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3446, 4.0426), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1427, 1.1696))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=educ*0.2138+eastwest*0.5294+6.7339, sigma=1.6968)
        income = pm.Normal('income', mu=tt.switch(tt.eq(sex, 0), educ*211.6522+eastwest*64.2077+happiness*31.6485+294.2901, educ*330.5615+eastwest*586.4313+happiness*160.9824+-645.3277), sigma=tt.switch(tt.eq(sex, 0), 787.7015, 1264.8692))
        age = pm.Normal('age', mu=educ*-5.1706+income*0.002+eastwest*-3.6209+69.1594, sigma=16.3352)
        health = pm.Normal('health', mu=age*-0.0166+educ*0.0905+income*0.0+happiness*0.2162+2.395, sigma=0.8371)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 37 parameter
#####################
def create_allbus_hc_predloglikcg(filename="", modelname="allbus_hc_predloglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), 3.3446, 4.0426), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.1427, 1.1696))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=educ*0.2138+eastwest*0.5294+6.7339, sigma=1.6968)
        income = pm.Normal('income', mu=tt.switch(tt.eq(sex, 0), educ*211.6522+eastwest*64.2077+happiness*31.6485+294.2901, educ*330.5615+eastwest*586.4313+happiness*160.9824+-645.3277), sigma=tt.switch(tt.eq(sex, 0), 787.7015, 1264.8692))
        age = pm.Normal('age', mu=educ*-5.1706+income*0.002+eastwest*-3.6209+69.1594, sigma=16.3352)
        health = pm.Normal('health', mu=age*-0.0166+educ*0.0905+income*0.0+happiness*0.2162+2.395, sigma=0.8371)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_gs_loglikcg(filename="", modelname="allbus_gs_loglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 44 parameter
#####################
def create_allbus_gs_aiccg(filename="", modelname="allbus_gs_aiccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        health = pm.Normal('health', mu=age*-0.0184+happiness*0.2311+2.7673, sigma=0.8464)
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0187+income*0.0003+health*0.1411+3.3888, age*-0.0136+income*0.0002+health*0.2281+3.5243), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0261, 1.0869))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 44 parameter
#####################
def create_allbus_gs_biccg(filename="", modelname="allbus_gs_biccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        health = pm.Normal('health', mu=age*-0.0184+happiness*0.2311+2.7673, sigma=0.8464)
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0187+income*0.0003+health*0.1411+3.3888, age*-0.0136+income*0.0002+health*0.2281+3.5243), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0261, 1.0869))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_gs_predloglikcg(filename="", modelname="allbus_gs_predloglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 44 parameter
#####################
def create_allbus_iamb_loglikcg(filename="", modelname="allbus_iamb_loglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        health = pm.Normal('health', mu=age*-0.0184+happiness*0.2311+2.7673, sigma=0.8464)
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0187+income*0.0003+health*0.1411+3.3888, age*-0.0136+income*0.0002+health*0.2281+3.5243), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0261, 1.0869))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_iamb_aiccg(filename="", modelname="allbus_iamb_aiccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 44 parameter
#####################
def create_allbus_iamb_biccg(filename="", modelname="allbus_iamb_biccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        health = pm.Normal('health', mu=age*-0.0184+happiness*0.2311+2.7673, sigma=0.8464)
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0187+income*0.0003+health*0.1411+3.3888, age*-0.0136+income*0.0002+health*0.2281+3.5243), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0261, 1.0869))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_iamb_predloglikcg(filename="", modelname="allbus_iamb_predloglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 44 parameter
#####################
def create_allbus_fastiamb_loglikcg(filename="", modelname="allbus_fastiamb_loglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        health = pm.Normal('health', mu=age*-0.0184+happiness*0.2311+2.7673, sigma=0.8464)
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0187+income*0.0003+health*0.1411+3.3888, age*-0.0136+income*0.0002+health*0.2281+3.5243), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0261, 1.0869))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_fastiamb_aiccg(filename="", modelname="allbus_fastiamb_aiccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_fastiamb_biccg(filename="", modelname="allbus_fastiamb_biccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_fastiamb_predloglikcg(filename="", modelname="allbus_fastiamb_predloglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 44 parameter
#####################
def create_allbus_interiamb_loglikcg(filename="", modelname="allbus_interiamb_loglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        health = pm.Normal('health', mu=age*-0.0184+happiness*0.2311+2.7673, sigma=0.8464)
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0187+income*0.0003+health*0.1411+3.3888, age*-0.0136+income*0.0002+health*0.2281+3.5243), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0261, 1.0869))
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_interiamb_aiccg(filename="", modelname="allbus_interiamb_aiccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_interiamb_biccg(filename="", modelname="allbus_interiamb_biccg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 43 parameter
#####################
def create_allbus_interiamb_predloglikcg(filename="", modelname="allbus_interiamb_predloglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803,0.5197])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145,0.1855])
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), 0.6306, 0.7896), sigma=tt.switch(tt.eq(lived_abroad, 0), 0.4828, 0.4081))
        happiness = pm.Normal('happiness', mu=eastwest*0.5501+7.4632, sigma=1.715)
        age = pm.Normal('age', mu=eastwest*-3.3054+54.5406, sigma=17.3657)
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), eastwest*64.7548+happiness*52.5931+830.6537, eastwest*559.0629+happiness*160.6072+449.146), tt.switch(tt.eq(sex, 0), eastwest*-24.8598+happiness*57.8588+1087.8252, eastwest*618.1169+happiness*306.519+-423.1545)), sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 778.8786, 1242.734), tt.switch(tt.eq(sex, 0), 995.2346, 1581.8897)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age*-0.0214+income*0.0003+4.0042, age*-0.0175+income*0.0002+4.538), sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0339, 1.108))
        health = pm.Normal('health', mu=age*-0.0162+educ*0.1047+happiness*0.2206+2.3698, sigma=0.8386)
    
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

