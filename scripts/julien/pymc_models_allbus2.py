#!usr/bin/python
# -*- coding: utf-8 -*-import string

import pymc3 as pm
import theano.tensor as tt

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper
from mb_modelbase.utils.Metrics import cll_allbus

import scripts.experiments.allbus as allbus_data

# LOAD FILES
test_data = allbus_data.train(discrete_happy=True)
train_data = allbus_data.test(discrete_happy=True)

df = train_data

# SAVE PARAMETER IN THIS FILE
model_file = 'allbus_results.dat'

sample_size = 100000

allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                      'lived_abroad': {'No': 0, 'Yes': 1},
                      'happiness': {'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8,
                                    'h9': 9, 'h10': 10}}

allbus_backward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                       'lived_abroad': {0: 'No', 1: 'Yes'},
                       'happiness': {0: 'h0', 1: 'h1', 2: 'h2', 3: 'h3', 4: 'h4', 5: 'h5', 6: 'h6', 7: 'h7', 8: 'h8',
                                     9: 'h9', 10: 'h10'}}

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)


#####################
# 64 parameter
#####################
def create_allbus_tabubiccg(filename="", modelname="allbus_tabubiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803, 0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347, 0.653])
        happiness = pm.Categorical('happiness',
                                   p=[0.0038, 0.0027, 0.0044, 0.0186, 0.0208, 0.0658, 0.0625, 0.1305, 0.3163, 0.2281,
                                      0.1464])
        health = pm.Normal('health', mu=tt.switch(tt.eq(happiness, 0), 1.8571, tt.switch(tt.eq(happiness, 1), 1.8,
                                                                                         tt.switch(tt.eq(happiness, 2),
                                                                                                   2.25, tt.switch(
                                                                                                 tt.eq(happiness, 3),
                                                                                                 3.0294, tt.switch(
                                                                                                     tt.eq(happiness,
                                                                                                           4), 2.5526,
                                                                                                     tt.switch(tt.eq(
                                                                                                         happiness, 5),
                                                                                                               2.8,
                                                                                                               tt.switch(
                                                                                                                   tt.eq(
                                                                                                                       happiness,
                                                                                                                       6),
                                                                                                                   3.2281,
                                                                                                                   tt.switch(
                                                                                                                       tt.eq(
                                                                                                                           happiness,
                                                                                                                           7),
                                                                                                                       3.3067,
                                                                                                                       tt.switch(
                                                                                                                           tt.eq(
                                                                                                                               happiness,
                                                                                                                               8),
                                                                                                                           3.669,
                                                                                                                           tt.switch(
                                                                                                                               tt.eq(
                                                                                                                                   happiness,
                                                                                                                                   9),
                                                                                                                               3.9736,
                                                                                                                               4.03)))))))))),
                           sigma=tt.switch(tt.eq(happiness, 0), 1.069, tt.switch(tt.eq(happiness, 1), 1.0954,
                                                                                 tt.switch(tt.eq(happiness, 2), 1.3887,
                                                                                           tt.switch(
                                                                                               tt.eq(happiness, 3),
                                                                                               1.0585, tt.switch(
                                                                                                   tt.eq(happiness, 4),
                                                                                                   1.1076, tt.switch(
                                                                                                       tt.eq(happiness,
                                                                                                             5), 1.058,
                                                                                                       tt.switch(tt.eq(
                                                                                                           happiness,
                                                                                                           6), 0.9124,
                                                                                                                 tt.switch(
                                                                                                                     tt.eq(
                                                                                                                         happiness,
                                                                                                                         7),
                                                                                                                     0.9292,
                                                                                                                     tt.switch(
                                                                                                                         tt.eq(
                                                                                                                             happiness,
                                                                                                                             8),
                                                                                                                         0.8247,
                                                                                                                         tt.switch(
                                                                                                                             tt.eq(
                                                                                                                                 happiness,
                                                                                                                                 9),
                                                                                                                             0.8625,
                                                                                                                             0.9374)))))))))))
        lived_abroad = pm.Categorical('lived_abroad',
                                      p=tt.switch(tt.eq(eastwest, 0), [0.8847, 0.1153], [0.7859, 0.2141]))
        educ = pm.Normal('educ',
                         mu=tt.switch(tt.eq(lived_abroad, 0), health * 0.3025 + 2.2566, health * 0.3004 + 2.919),
                         sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0995, 1.1302))
        income = pm.Normal('income', mu=tt.switch(tt.eq(eastwest, 0),
                                                  tt.switch(tt.eq(sex, 0), educ * 170.0602 + 665.7868,
                                                            educ * 359.6632 + 460.3479),
                                                  tt.switch(tt.eq(sex, 0), educ * 220.2511 + 593.4239,
                                                            educ * 409.6346 + 993.8322)),
                           sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 649.0034, 975.7261),
                                           tt.switch(tt.eq(sex, 0), 844.5987, 1470.5398)))
        age = pm.Normal('age', mu=educ * -4.2228 + income * 0.0021 + health * -4.6853 + 80.3168, sigma=15.8285)

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m


#####################
# 187 parameter
#####################
def create_allbus_tabuaiccg(filename="", modelname="allbus_tabuaiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8202, 0.1798])
        eastwest = pm.Categorical('eastwest', p=tt.switch(tt.eq(lived_abroad, 0), [0.3743, 0.6257], [0.2226, 0.7774]))
        happiness = pm.Categorical('happiness', p=tt.switch(tt.eq(eastwest, 0),
                                                            [0.0032, 0.0032, 0.0079, 0.0269, 0.019, 0.09, 0.0821,
                                                             0.1675, 0.3333, 0.1896, 0.0774],
                                                            [0.0042, 0.0025, 0.0025, 0.0143, 0.0218, 0.0529, 0.0521,
                                                             0.1108, 0.3073, 0.2485, 0.183]))
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(happiness, 0), [0.4286, 0.5714],
                                                tt.switch(tt.eq(happiness, 1), [1.0, 0.0],
                                                          tt.switch(tt.eq(happiness, 2), [0.375, 0.625],
                                                                    tt.switch(tt.eq(happiness, 3), [0.2941, 0.7059],
                                                                              tt.switch(tt.eq(happiness, 4),
                                                                                        [0.4211, 0.5789],
                                                                                        tt.switch(tt.eq(happiness, 5),
                                                                                                  [0.5, 0.5], tt.switch(
                                                                                                tt.eq(happiness, 6),
                                                                                                [0.4737, 0.5263],
                                                                                                tt.switch(
                                                                                                    tt.eq(happiness, 7),
                                                                                                    [0.4118, 0.5882],
                                                                                                    tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              8),
                                                                                                        [0.4541,
                                                                                                         0.5459],
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            9), [0.5312,
                                                                                                                 0.4688],
                                                                                                                  [
                                                                                                                      0.5393,
                                                                                                                      0.4607])))))))))))
        income = pm.Normal('income', mu=tt.switch(tt.eq(happiness, 0), tt.switch(tt.eq(sex, 0), 636.6667, 991.0),
                                                  tt.switch(tt.eq(happiness, 1), tt.switch(tt.eq(sex, 0), 854.6, 0.0),
                                                            tt.switch(tt.eq(happiness, 2),
                                                                      tt.switch(tt.eq(sex, 0), 923.3333, 1091.6),
                                                                      tt.switch(tt.eq(happiness, 3),
                                                                                tt.switch(tt.eq(sex, 0), 1093.2,
                                                                                          1246.7917),
                                                                                tt.switch(tt.eq(happiness, 4),
                                                                                          tt.switch(tt.eq(sex, 0),
                                                                                                    1122.8125, 1372.5),
                                                                                          tt.switch(tt.eq(happiness, 5),
                                                                                                    tt.switch(
                                                                                                        tt.eq(sex, 0),
                                                                                                        1093.0667,
                                                                                                        1423.9333),
                                                                                                    tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              6),
                                                                                                        tt.switch(
                                                                                                            tt.eq(sex,
                                                                                                                  0),
                                                                                                            1213.0185,
                                                                                                            1508.2),
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            7),
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          sex,
                                                                                                                          0),
                                                                                                                      1256.602,
                                                                                                                      1885.4929),
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          happiness,
                                                                                                                          8),
                                                                                                                      tt.switch(
                                                                                                                          tt.eq(
                                                                                                                              sex,
                                                                                                                              0),
                                                                                                                          1415.2328,
                                                                                                                          2281.0698),
                                                                                                                      tt.switch(
                                                                                                                          tt.eq(
                                                                                                                              happiness,
                                                                                                                              9),
                                                                                                                          tt.switch(
                                                                                                                              tt.eq(
                                                                                                                                  sex,
                                                                                                                                  0),
                                                                                                                              1424.3846,
                                                                                                                              2675.0051),
                                                                                                                          tt.switch(
                                                                                                                              tt.eq(
                                                                                                                                  sex,
                                                                                                                                  0),
                                                                                                                              1317.3819,
                                                                                                                              2322.9675))))))))))),
                           sigma=tt.switch(tt.eq(happiness, 0), tt.switch(tt.eq(sex, 0), 403.7739, 367.6284),
                                           tt.switch(tt.eq(happiness, 1), tt.switch(tt.eq(sex, 0), 228.8904, 0.0),
                                                     tt.switch(tt.eq(happiness, 2),
                                                               tt.switch(tt.eq(sex, 0), 601.3596, 651.9761),
                                                               tt.switch(tt.eq(happiness, 3),
                                                                         tt.switch(tt.eq(sex, 0), 528.3283, 667.7649),
                                                                         tt.switch(tt.eq(happiness, 4),
                                                                                   tt.switch(tt.eq(sex, 0), 554.5323,
                                                                                             695.9161),
                                                                                   tt.switch(tt.eq(happiness, 5),
                                                                                             tt.switch(tt.eq(sex, 0),
                                                                                                       529.4652,
                                                                                                       971.2261),
                                                                                             tt.switch(
                                                                                                 tt.eq(happiness, 6),
                                                                                                 tt.switch(
                                                                                                     tt.eq(sex, 0),
                                                                                                     695.1082,
                                                                                                     919.7047),
                                                                                                 tt.switch(
                                                                                                     tt.eq(happiness,
                                                                                                           7),
                                                                                                     tt.switch(
                                                                                                         tt.eq(sex, 0),
                                                                                                         688.9785,
                                                                                                         1143.9423),
                                                                                                     tt.switch(tt.eq(
                                                                                                         happiness, 8),
                                                                                                               tt.switch(
                                                                                                                   tt.eq(
                                                                                                                       sex,
                                                                                                                       0),
                                                                                                                   850.6288,
                                                                                                                   1347.8284),
                                                                                                               tt.switch(
                                                                                                                   tt.eq(
                                                                                                                       happiness,
                                                                                                                       9),
                                                                                                                   tt.switch(
                                                                                                                       tt.eq(
                                                                                                                           sex,
                                                                                                                           0),
                                                                                                                       901.945,
                                                                                                                       1744.6562),
                                                                                                                   tt.switch(
                                                                                                                       tt.eq(
                                                                                                                           sex,
                                                                                                                           0),
                                                                                                                       887.7215,
                                                                                                                       1617.5122))))))))))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(eastwest, 0), income * -0.0018 + 57.3971, income * 0.0012 + 48.9231),
                        sigma=tt.switch(tt.eq(eastwest, 0), 17.1271, 17.4177))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0),
                                                                                tt.switch(tt.eq(sex, 0),
                                                                                          age * -0.0236 + income * 0.0005 + 4.087,
                                                                                          age * -0.0092 + income * 0.0004 + 3.0935),
                                                                                tt.switch(tt.eq(sex, 0),
                                                                                          age * -0.0272 + income * 0.0005 + 4.1636,
                                                                                          age * -0.0211 + income * 0.0003 + 3.7571)),
                                              tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0),
                                                                                      age * -0.0439 + income * 0.0001 + 6.191,
                                                                                      age * 0.0019 + income * 0.0001 + 3.8121),
                                                        tt.switch(tt.eq(sex, 0),
                                                                  age * -0.016 + income * 0.0002 + 4.7033,
                                                                  age * -0.0194 + income * 0.0002 + 4.2361))),
                         sigma=tt.switch(tt.eq(lived_abroad, 0),
                                         tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.9173, 0.9503),
                                                   tt.switch(tt.eq(sex, 0), 1.0124, 1.0879)),
                                         tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.8676, 1.1247),
                                                   tt.switch(tt.eq(sex, 0), 1.0018, 1.1577))))
        health = pm.Normal('health',
                           mu=tt.switch(tt.eq(happiness, 0), age * 0.0164 + educ * -0.613 + income * 0.002 + 0.7467,
                                        tt.switch(tt.eq(happiness, 1),
                                                  age * -0.0394 + educ * -0.6432 + income * 0.0037 + 2.3491,
                                                  tt.switch(tt.eq(happiness, 2),
                                                            age * 0.0593 + educ * 0.0521 + income * 0.0024 + -3.1963,
                                                            tt.switch(tt.eq(happiness, 3),
                                                                      age * -0.0048 + educ * 0.0558 + income * 0.0007 + 2.2155,
                                                                      tt.switch(tt.eq(happiness, 4),
                                                                                age * -0.02 + educ * 0.0463 + income * 0.0006 + 2.6712,
                                                                                tt.switch(tt.eq(happiness, 5),
                                                                                          age * -0.0173 + educ * 0.009 + income * 0.0001 + 3.6444,
                                                                                          tt.switch(tt.eq(happiness, 6),
                                                                                                    age * -0.0113 + educ * 0.112 + income * 0.0 + 3.5495,
                                                                                                    tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              7),
                                                                                                        age * -0.0203 + educ * 0.0006 + income * 0.0001 + 4.2444,
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            8),
                                                                                                                  age * -0.0152 + educ * 0.0917 + income * 0.0 + 4.0633,
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          happiness,
                                                                                                                          9),
                                                                                                                      age * -0.0171 + educ * 0.115 + income * 0.0001 + 4.3001,
                                                                                                                      age * -0.0156 + educ * 0.1568 + income * 0.0 + 4.3034)))))))))),
                           sigma=tt.switch(tt.eq(happiness, 0), 0.3772, tt.switch(tt.eq(happiness, 1), 0.0951,
                                                                                  tt.switch(tt.eq(happiness, 2), 1.2401,
                                                                                            tt.switch(
                                                                                                tt.eq(happiness, 3),
                                                                                                0.9973, tt.switch(
                                                                                                    tt.eq(happiness, 4),
                                                                                                    1.0111, tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              5),
                                                                                                        1.0245,
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            6), 0.8813,
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          happiness,
                                                                                                                          7),
                                                                                                                      0.8664,
                                                                                                                      tt.switch(
                                                                                                                          tt.eq(
                                                                                                                              happiness,
                                                                                                                              8),
                                                                                                                          0.763,
                                                                                                                          tt.switch(
                                                                                                                              tt.eq(
                                                                                                                                  happiness,
                                                                                                                                  9),
                                                                                                                              0.7836,
                                                                                                                              0.8502)))))))))))

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m


#####################
# 64 parameter
#####################
def create_allbus_hcbiccg(filename="", modelname="allbus_hcbiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        sex = pm.Categorical('sex', p=[0.4803, 0.5197])
        eastwest = pm.Categorical('eastwest', p=[0.347, 0.653])
        happiness = pm.Categorical('happiness',
                                   p=[0.0038, 0.0027, 0.0044, 0.0186, 0.0208, 0.0658, 0.0625, 0.1305, 0.3163, 0.2281,
                                      0.1464])
        health = pm.Normal('health', mu=tt.switch(tt.eq(happiness, 0), 1.8571, tt.switch(tt.eq(happiness, 1), 1.8,
                                                                                         tt.switch(tt.eq(happiness, 2),
                                                                                                   2.25, tt.switch(
                                                                                                 tt.eq(happiness, 3),
                                                                                                 3.0294, tt.switch(
                                                                                                     tt.eq(happiness,
                                                                                                           4), 2.5526,
                                                                                                     tt.switch(tt.eq(
                                                                                                         happiness, 5),
                                                                                                               2.8,
                                                                                                               tt.switch(
                                                                                                                   tt.eq(
                                                                                                                       happiness,
                                                                                                                       6),
                                                                                                                   3.2281,
                                                                                                                   tt.switch(
                                                                                                                       tt.eq(
                                                                                                                           happiness,
                                                                                                                           7),
                                                                                                                       3.3067,
                                                                                                                       tt.switch(
                                                                                                                           tt.eq(
                                                                                                                               happiness,
                                                                                                                               8),
                                                                                                                           3.669,
                                                                                                                           tt.switch(
                                                                                                                               tt.eq(
                                                                                                                                   happiness,
                                                                                                                                   9),
                                                                                                                               3.9736,
                                                                                                                               4.03)))))))))),
                           sigma=tt.switch(tt.eq(happiness, 0), 1.069, tt.switch(tt.eq(happiness, 1), 1.0954,
                                                                                 tt.switch(tt.eq(happiness, 2), 1.3887,
                                                                                           tt.switch(
                                                                                               tt.eq(happiness, 3),
                                                                                               1.0585, tt.switch(
                                                                                                   tt.eq(happiness, 4),
                                                                                                   1.1076, tt.switch(
                                                                                                       tt.eq(happiness,
                                                                                                             5), 1.058,
                                                                                                       tt.switch(tt.eq(
                                                                                                           happiness,
                                                                                                           6), 0.9124,
                                                                                                                 tt.switch(
                                                                                                                     tt.eq(
                                                                                                                         happiness,
                                                                                                                         7),
                                                                                                                     0.9292,
                                                                                                                     tt.switch(
                                                                                                                         tt.eq(
                                                                                                                             happiness,
                                                                                                                             8),
                                                                                                                         0.8247,
                                                                                                                         tt.switch(
                                                                                                                             tt.eq(
                                                                                                                                 happiness,
                                                                                                                                 9),
                                                                                                                             0.8625,
                                                                                                                             0.9374)))))))))))
        lived_abroad = pm.Categorical('lived_abroad',
                                      p=tt.switch(tt.eq(eastwest, 0), [0.8847, 0.1153], [0.7859, 0.2141]))
        educ = pm.Normal('educ',
                         mu=tt.switch(tt.eq(lived_abroad, 0), health * 0.3025 + 2.2566, health * 0.3004 + 2.919),
                         sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0995, 1.1302))
        income = pm.Normal('income', mu=tt.switch(tt.eq(eastwest, 0),
                                                  tt.switch(tt.eq(sex, 0), educ * 170.0602 + 665.7868,
                                                            educ * 359.6632 + 460.3479),
                                                  tt.switch(tt.eq(sex, 0), educ * 220.2511 + 593.4239,
                                                            educ * 409.6346 + 993.8322)),
                           sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 649.0034, 975.7261),
                                           tt.switch(tt.eq(sex, 0), 844.5987, 1470.5398)))
        age = pm.Normal('age', mu=educ * -4.2228 + income * 0.0021 + health * -4.6853 + 80.3168, sigma=15.8285)

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m


#####################
# 197 parameter
#####################
def create_allbus_hcaiccg(filename="", modelname="allbus_hcaiccg", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        eastwest = pm.Categorical('eastwest', p=[0.347, 0.653])
        happiness = pm.Categorical('happiness', p=tt.switch(tt.eq(eastwest, 0),
                                                            [0.0032, 0.0032, 0.0079, 0.0269, 0.019, 0.09, 0.0821,
                                                             0.1675, 0.3333, 0.1896, 0.0774],
                                                            [0.0042, 0.0025, 0.0025, 0.0143, 0.0218, 0.0529, 0.0521,
                                                             0.1108, 0.3073, 0.2485, 0.183]))
        lived_abroad = pm.Categorical('lived_abroad',
                                      p=tt.switch(tt.eq(eastwest, 0), [0.8847, 0.1153], [0.7859, 0.2141]))
        age = pm.Normal('age', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), 55.2714, 51.7959),
                                            tt.switch(tt.eq(eastwest, 0), 51.0548, 48.9451)),
                        sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0), 17.0513, 17.4567),
                                        tt.switch(tt.eq(eastwest, 0), 17.9086, 17.4475)))
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(happiness, 0), [0.4286, 0.5714],
                                                tt.switch(tt.eq(happiness, 1), [1.0, 0.0],
                                                          tt.switch(tt.eq(happiness, 2), [0.375, 0.625],
                                                                    tt.switch(tt.eq(happiness, 3), [0.2941, 0.7059],
                                                                              tt.switch(tt.eq(happiness, 4),
                                                                                        [0.4211, 0.5789],
                                                                                        tt.switch(tt.eq(happiness, 5),
                                                                                                  [0.5, 0.5], tt.switch(
                                                                                                tt.eq(happiness, 6),
                                                                                                [0.4737, 0.5263],
                                                                                                tt.switch(
                                                                                                    tt.eq(happiness, 7),
                                                                                                    [0.4118, 0.5882],
                                                                                                    tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              8),
                                                                                                        [0.4541,
                                                                                                         0.5459],
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            9), [0.5312,
                                                                                                                 0.4688],
                                                                                                                  [
                                                                                                                      0.5393,
                                                                                                                      0.4607])))))))))))
        health = pm.Normal('health', mu=tt.switch(tt.eq(happiness, 0), age * 0.0378 + -0.2158,
                                                  tt.switch(tt.eq(happiness, 1), age * 0.0287 + 0.3807,
                                                            tt.switch(tt.eq(happiness, 2), age * -0.0491 + 4.588,
                                                                      tt.switch(tt.eq(happiness, 3),
                                                                                age * -0.0011 + 3.0831,
                                                                                tt.switch(tt.eq(happiness, 4),
                                                                                          age * -0.0206 + 3.6572,
                                                                                          tt.switch(tt.eq(happiness, 5),
                                                                                                    age * -0.018 + 3.862,
                                                                                                    tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              6),
                                                                                                        age * -0.0138 + 3.9815,
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            7),
                                                                                                                  age * -0.0201 + 4.3399,
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          happiness,
                                                                                                                          8),
                                                                                                                      age * -0.0171 + 4.5497,
                                                                                                                      tt.switch(
                                                                                                                          tt.eq(
                                                                                                                              happiness,
                                                                                                                              9),
                                                                                                                          age * -0.0191 + 4.9273,
                                                                                                                          age * -0.0192 + 5.0995)))))))))),
                           sigma=tt.switch(tt.eq(happiness, 0), 0.8806, tt.switch(tt.eq(happiness, 1), 1.2092,
                                                                                  tt.switch(tt.eq(happiness, 2), 1.3859,
                                                                                            tt.switch(
                                                                                                tt.eq(happiness, 3),
                                                                                                1.0747, tt.switch(
                                                                                                    tt.eq(happiness, 4),
                                                                                                    1.0757, tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              5),
                                                                                                        1.0206,
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            6), 0.8811,
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          happiness,
                                                                                                                          7),
                                                                                                                      0.8654,
                                                                                                                      tt.switch(
                                                                                                                          tt.eq(
                                                                                                                              happiness,
                                                                                                                              8),
                                                                                                                          0.7712,
                                                                                                                          tt.switch(
                                                                                                                              tt.eq(
                                                                                                                                  happiness,
                                                                                                                                  9),
                                                                                                                              0.7991,
                                                                                                                              0.8693)))))))))))
        income = pm.Normal('income', mu=tt.switch(tt.eq(happiness, 0), tt.switch(tt.eq(sex, 0), health * 347.5 + 57.5,
                                                                                 health * 309.0 + 373.0),
                                                  tt.switch(tt.eq(happiness, 1),
                                                            tt.switch(tt.eq(sex, 0), health * 183.6667 + 524.0,
                                                                      health * 0.0 + 0.0),
                                                            tt.switch(tt.eq(happiness, 2), tt.switch(tt.eq(sex, 0),
                                                                                                     health * 575.0 + -226.6667,
                                                                                                     health * 240.7857 + 513.7143),
                                                                      tt.switch(tt.eq(happiness, 3),
                                                                                tt.switch(tt.eq(sex, 0),
                                                                                          health * -42.6939 + 1182.8571,
                                                                                          health * 414.1555 + -168.2395),
                                                                                tt.switch(tt.eq(happiness, 4),
                                                                                          tt.switch(tt.eq(sex, 0),
                                                                                                    health * -36.4068 + 1202.4525,
                                                                                                    health * 370.795 + 327.5324),
                                                                                          tt.switch(tt.eq(happiness, 5),
                                                                                                    tt.switch(
                                                                                                        tt.eq(sex, 0),
                                                                                                        health * 135.3537 + 716.3322,
                                                                                                        health * 34.6745 + 1326.2667),
                                                                                                    tt.switch(
                                                                                                        tt.eq(happiness,
                                                                                                              6),
                                                                                                        tt.switch(
                                                                                                            tt.eq(sex,
                                                                                                                  0),
                                                                                                            health * -98.8618 + 1535.2348,
                                                                                                            health * 54.5084 + 1333.7731),
                                                                                                        tt.switch(tt.eq(
                                                                                                            happiness,
                                                                                                            7),
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          sex,
                                                                                                                          0),
                                                                                                                      health * 7.851 + 1230.0047,
                                                                                                                      health * 127.1462 + 1472.2676),
                                                                                                                  tt.switch(
                                                                                                                      tt.eq(
                                                                                                                          happiness,
                                                                                                                          8),
                                                                                                                      tt.switch(
                                                                                                                          tt.eq(
                                                                                                                              sex,
                                                                                                                              0),
                                                                                                                          health * 107.5738 + 1016.1421,
                                                                                                                          health * 177.7213 + 1635.067),
                                                                                                                      tt.switch(
                                                                                                                          tt.eq(
                                                                                                                              happiness,
                                                                                                                              9),
                                                                                                                          tt.switch(
                                                                                                                              tt.eq(
                                                                                                                                  sex,
                                                                                                                                  0),
                                                                                                                              health * 129.6497 + 917.5187,
                                                                                                                              health * 69.0046 + 2395.8017),
                                                                                                                          tt.switch(
                                                                                                                              tt.eq(
                                                                                                                                  sex,
                                                                                                                                  0),
                                                                                                                              health * 71.4611 + 1025.5824,
                                                                                                                              health * 249.7763 + 1331.9852))))))))))),
                           sigma=tt.switch(tt.eq(happiness, 0), tt.switch(tt.eq(sex, 0), 63.6396, 108.462),
                                           tt.switch(tt.eq(happiness, 1), tt.switch(tt.eq(sex, 0), 126.0194, 0.0),
                                                     tt.switch(tt.eq(happiness, 2),
                                                               tt.switch(tt.eq(sex, 0), 249.0315, 591.8728),
                                                               tt.switch(tt.eq(happiness, 3),
                                                                         tt.switch(tt.eq(sex, 0), 559.3798, 558.1635),
                                                                         tt.switch(tt.eq(happiness, 4),
                                                                                   tt.switch(tt.eq(sex, 0), 572.6382,
                                                                                             578.6002),
                                                                                   tt.switch(tt.eq(happiness, 5),
                                                                                             tt.switch(tt.eq(sex, 0),
                                                                                                       513.4486,
                                                                                                       978.8751),
                                                                                             tt.switch(
                                                                                                 tt.eq(happiness, 6),
                                                                                                 tt.switch(
                                                                                                     tt.eq(sex, 0),
                                                                                                     695.5226, 926.284),
                                                                                                 tt.switch(
                                                                                                     tt.eq(happiness,
                                                                                                           7),
                                                                                                     tt.switch(
                                                                                                         tt.eq(sex, 0),
                                                                                                         692.5172,
                                                                                                         1142.1333),
                                                                                                     tt.switch(tt.eq(
                                                                                                         happiness, 8),
                                                                                                               tt.switch(
                                                                                                                   tt.eq(
                                                                                                                       sex,
                                                                                                                       0),
                                                                                                                   847.0783,
                                                                                                                   1342.7465),
                                                                                                               tt.switch(
                                                                                                                   tt.eq(
                                                                                                                       happiness,
                                                                                                                       9),
                                                                                                                   tt.switch(
                                                                                                                       tt.eq(
                                                                                                                           sex,
                                                                                                                           0),
                                                                                                                       896.5767,
                                                                                                                       1748.2349),
                                                                                                                   tt.switch(
                                                                                                                       tt.eq(
                                                                                                                           sex,
                                                                                                                           0),
                                                                                                                       888.7199,
                                                                                                                       1603.7575))))))))))))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(eastwest, 0),
                                                                                tt.switch(tt.eq(sex, 0),
                                                                                          age * -0.0235 + income * 0.0005 + health * 0.0013 + 4.0819,
                                                                                          age * -0.0071 + income * 0.0004 + health * 0.1178 + 2.6051),
                                                                                tt.switch(tt.eq(sex, 0),
                                                                                          age * -0.0238 + income * 0.0004 + health * 0.1837 + 3.3505,
                                                                                          age * -0.0176 + income * 0.0003 + health * 0.1701 + 3.0179)),
                                              tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0),
                                                                                      age * -0.0389 + income * 0.0 + health * 0.4528 + 4.4224,
                                                                                      age * 0.0135 + income * 8.6743e-06 + health * 0.4529 + 1.6102),
                                                        tt.switch(tt.eq(sex, 0),
                                                                  age * -0.0142 + income * 0.0002 + health * 0.1889 + 3.8769,
                                                                  age * -0.017 + income * 0.0002 + health * 0.1371 + 3.6386))),
                         sigma=tt.switch(tt.eq(lived_abroad, 0),
                                         tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.919, 0.9464),
                                                   tt.switch(tt.eq(sex, 0), 0.9982, 1.0788)),
                                         tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(sex, 0), 0.7676, 1.0763),
                                                   tt.switch(tt.eq(sex, 0), 0.9882, 1.1537))))

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m


#####################
# 56 parameter
#####################
def create_allbus_gs(filename="", modelname="allbus_gs", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=52.4348, sigma=17.464)
        sex = pm.Categorical('sex', p=[0.4803, 0.5197])
        happiness = pm.Categorical('happiness',
                                   p=[0.0038, 0.0027, 0.0044, 0.0186, 0.0208, 0.0658, 0.0625, 0.1305, 0.3163, 0.2281,
                                      0.1464])
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8202, 0.1798])
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), 1679.7861, 2102.8415),
                           sigma=tt.switch(tt.eq(lived_abroad, 0), 1156.5519, 1570.8752))
        eastwest = pm.Categorical('eastwest', p=tt.switch(tt.eq(happiness, 0), [0.2857, 0.7143],
                                                          tt.switch(tt.eq(happiness, 1), [0.4, 0.6],
                                                                    tt.switch(tt.eq(happiness, 2), [0.625, 0.375],
                                                                              tt.switch(tt.eq(happiness, 3), [0.5, 0.5],
                                                                                        tt.switch(tt.eq(happiness, 4),
                                                                                                  [0.3158, 0.6842],
                                                                                                  tt.switch(
                                                                                                      tt.eq(happiness,
                                                                                                            5),
                                                                                                      [0.475, 0.525],
                                                                                                      tt.switch(tt.eq(
                                                                                                          happiness, 6),
                                                                                                                [0.4561,
                                                                                                                 0.5439],
                                                                                                                tt.switch(
                                                                                                                    tt.eq(
                                                                                                                        happiness,
                                                                                                                        7),
                                                                                                                    [
                                                                                                                        0.4454,
                                                                                                                        0.5546],
                                                                                                                    tt.switch(
                                                                                                                        tt.eq(
                                                                                                                            happiness,
                                                                                                                            8),
                                                                                                                        [
                                                                                                                            0.3657,
                                                                                                                            0.6343],
                                                                                                                        tt.switch(
                                                                                                                            tt.eq(
                                                                                                                                happiness,
                                                                                                                                9),
                                                                                                                            [
                                                                                                                                0.2885,
                                                                                                                                0.7115],
                                                                                                                            [
                                                                                                                                0.1835,
                                                                                                                                0.8165])))))))))))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), age * -0.0211 + income * 0.0003 + 3.9633,
                                              age * -0.0194 + income * 0.0002 + 4.7036),
                         sigma=tt.switch(tt.eq(lived_abroad, 0), 1.0284, 1.1018))
        health = pm.Normal('health', mu=age * -0.0157 + educ * 0.1297 + income * 0.0001 + 3.8148, sigma=0.9174)

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m


#####################
# 43 parameter
#####################
def create_allbus_iamb(filename="", modelname="allbus_iamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=52.4348, sigma=17.464)
        sex = pm.Categorical('sex', p=[0.4803, 0.5197])
        income = pm.Normal('income', mu=1755.8618, sigma=1251.3951)
        eastwest = pm.Categorical('eastwest', p=[0.347, 0.653])
        happiness = pm.Categorical('happiness', p=tt.switch(tt.eq(eastwest, 0),
                                                            [0.0032, 0.0032, 0.0079, 0.0269, 0.019, 0.09, 0.0821,
                                                             0.1675, 0.3333, 0.1896, 0.0774],
                                                            [0.0042, 0.0025, 0.0025, 0.0143, 0.0218, 0.0529, 0.0521,
                                                             0.1108, 0.3073, 0.2485, 0.183]))
        lived_abroad = pm.Categorical('lived_abroad',
                                      p=tt.switch(tt.eq(eastwest, 0), [0.8847, 0.1153], [0.7859, 0.2141]))
        educ = pm.Normal('educ', mu=age * -0.0218 + income * 0.0003 + 4.1308, sigma=1.0642)
        health = pm.Normal('health', mu=age * -0.0157 + educ * 0.1297 + income * 0.0001 + 3.8148, sigma=0.9174)

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m


#####################
# 48 parameter
#####################
def create_allbus_fastiamb(filename="", modelname="allbus_fastiamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=52.4348, sigma=17.464)
        sex = pm.Categorical('sex', p=[0.4803, 0.5197])
        educ = pm.Normal('educ', mu=age * -0.0213 + 4.5826, sigma=1.1175)
        happiness = pm.Categorical('happiness',
                                   p=[0.0038, 0.0027, 0.0044, 0.0186, 0.0208, 0.0658, 0.0625, 0.1305, 0.3163, 0.2281,
                                      0.1464])
        health = pm.Normal('health', mu=3.6058, sigma=0.9941)
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8202, 0.1798])
        income = pm.Normal('income', mu=educ * 274.5168 + health * 106.2597 + 421.5328, sigma=1197.1419)
        eastwest = pm.Categorical('eastwest', p=tt.switch(tt.eq(happiness, 0), [0.2857, 0.7143],
                                                          tt.switch(tt.eq(happiness, 1), [0.4, 0.6],
                                                                    tt.switch(tt.eq(happiness, 2), [0.625, 0.375],
                                                                              tt.switch(tt.eq(happiness, 3), [0.5, 0.5],
                                                                                        tt.switch(tt.eq(happiness, 4),
                                                                                                  [0.3158, 0.6842],
                                                                                                  tt.switch(
                                                                                                      tt.eq(happiness,
                                                                                                            5),
                                                                                                      [0.475, 0.525],
                                                                                                      tt.switch(tt.eq(
                                                                                                          happiness, 6),
                                                                                                                [0.4561,
                                                                                                                 0.5439],
                                                                                                                tt.switch(
                                                                                                                    tt.eq(
                                                                                                                        happiness,
                                                                                                                        7),
                                                                                                                    [
                                                                                                                        0.4454,
                                                                                                                        0.5546],
                                                                                                                    tt.switch(
                                                                                                                        tt.eq(
                                                                                                                            happiness,
                                                                                                                            8),
                                                                                                                        [
                                                                                                                            0.3657,
                                                                                                                            0.6343],
                                                                                                                        tt.switch(
                                                                                                                            tt.eq(
                                                                                                                                happiness,
                                                                                                                                9),
                                                                                                                            [
                                                                                                                                0.2885,
                                                                                                                                0.7115],
                                                                                                                            [
                                                                                                                                0.1835,
                                                                                                                                0.8165])))))))))))

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m


#####################
# 65 parameter
#####################
def create_allbus_interiamb(filename="", modelname="allbus_interiamb", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=52.4348, sigma=17.464)
        sex = pm.Categorical('sex', p=[0.4803, 0.5197])
        income = pm.Normal('income', mu=1755.8618, sigma=1251.3951)
        eastwest = pm.Categorical('eastwest', p=[0.347, 0.653])
        happiness = pm.Categorical('happiness', p=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(eastwest, 0),
                                                                                     [0.0034, 0.0068, 0.0068, 0.0101,
                                                                                      0.0203, 0.0946, 0.0845, 0.1588,
                                                                                      0.2905, 0.2331, 0.0912],
                                                                                     [0.0034, 0.0052, 0.0017, 0.0121,
                                                                                      0.0172, 0.0552, 0.05, 0.0879,
                                                                                      0.3034, 0.2621, 0.2017]),
                                                            tt.switch(tt.eq(eastwest, 0),
                                                                      [0.003, 0.0, 0.0089, 0.0415, 0.0178, 0.0861,
                                                                       0.0801, 0.1751, 0.3709, 0.1513, 0.0653],
                                                                      [0.0049, 0.0, 0.0033, 0.0164, 0.0262, 0.0507,
                                                                       0.054, 0.1326, 0.311, 0.2357, 0.1653])))
        health = pm.Normal('health', mu=age * -0.0186 + income * 0.0001 + 4.3506, sigma=0.9275)
        lived_abroad = pm.Categorical('lived_abroad',
                                      p=tt.switch(tt.eq(eastwest, 0), [0.8847, 0.1153], [0.7859, 0.2141]))
        educ = pm.Normal('educ', mu=age * -0.0187 + income * 0.0003 + health * 0.1708 + 3.3879, sigma=1.0526)

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        cll_allbus(m, test_data, model_file)
    return df, m

