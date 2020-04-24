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

allbus_forward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                      'lived_abroad': {0: 'No', 1: 'Yes'}
                      }

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)

    #####################
    # 114 parameter
    #####################


def create_allbus_tabu_loglikcg(filename="", modelname="allbus_tabu_loglikcg", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
        lived_abroad = pm.Categorical('lived_abroad', p=[0.8145, 0.1855])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(lived_abroad, 0), [0.4863, 0.5137], [0.4539, 0.5461]))
        eastwest = pm.Normal('eastwest', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.6445, 0.6174),
                                                      tt.switch(tt.eq(sex, 0), 0.8021, 0.7792)),
                             sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.4789, 0.4863),
                                             tt.switch(tt.eq(sex, 0), 0.3995, 0.4157)))
        happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(lived_abroad, 0),
                                                        tt.switch(tt.eq(sex, 0), eastwest * 0.4998 + 7.5483,
                                                                  eastwest * 0.5703 + 7.3397),
                                                        tt.switch(tt.eq(sex, 0), eastwest * 0.6012 + 7.7105,
                                                                  eastwest * 0.3559 + 7.6275)),
                              sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.7528, 1.7169),
                                              tt.switch(tt.eq(sex, 0), 1.5438, 1.6729)))
        health = pm.Normal('health', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0),
                                                                                    eastwest * 0.1351 + happiness * 0.2455 + 1.5642,
                                                                                    eastwest * -0.0133 + happiness * 0.1928 + 2.0811),
                                                  tt.switch(tt.eq(sex, 0),
                                                            eastwest * -0.0037 + happiness * 0.303 + 1.4271,
                                                            eastwest * -0.1941 + happiness * 0.269 + 1.7522)),
                           sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 0.9097, 0.8876),
                                           tt.switch(tt.eq(sex, 0), 0.8722, 0.9391)))
        income = pm.Normal('income', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0),
                                                                                    eastwest * 52.4719 + happiness * 30.2713 + health * 90.9186 + 688.4361,
                                                                                    eastwest * 560.8984 + happiness * 133.9881 + health * 138.0922 + 161.7676),
                                                  tt.switch(tt.eq(sex, 0),
                                                            eastwest * -24.9051 + happiness * 61.542 + health * -12.157 + 1105.1747,
                                                            eastwest * 646.4794 + happiness * 267.212 + health * 146.1254 + -679.192)),
                           sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 774.9052, 1237.3257),
                                           tt.switch(tt.eq(sex, 0), 997.8213, 1579.3942)))
        educ = pm.Normal('educ', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0),
                                                                                income * 0.0004 + eastwest * -0.0857 + happiness * 0.0471 + health * 0.1932 + 1.8236,
                                                                                income * 0.0002 + eastwest * -0.1542 + happiness * -0.0119 + health * 0.2758 + 2.0366),
                                              tt.switch(tt.eq(sex, 0),
                                                        income * 0.0002 + eastwest * 0.027 + happiness * -0.086 + health * 0.3679 + 3.1361,
                                                        income * 0.0002 + eastwest * -0.1625 + happiness * -0.0401 + health * 0.2618 + 2.87)),
                         sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 1.0447, 1.0748),
                                         tt.switch(tt.eq(sex, 0), 1.0249, 1.1456)))
        age = pm.Normal('age', mu=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0),
                                                                              educ * -5.8253 + income * 0.0032 + eastwest * -3.9383 + happiness * 1.8691 + health * -5.0885 + 74.4119,
                                                                              educ * -3.4663 + income * 0.0024 + eastwest * -3.9209 + happiness * 1.0883 + health * -6.8246 + 78.4069),
                                            tt.switch(tt.eq(sex, 0),
                                                      educ * -5.4212 + income * 0.0009 + eastwest * -3.7137 + happiness * 2.9171 + health * -4.5003 + 64.6867,
                                                      educ * -1.0406 + income * 0.0015 + eastwest * -0.2426 + happiness * 0.9936 + health * -5.8061 + 65.1705)),
                        sigma=tt.switch(tt.eq(lived_abroad, 0), tt.switch(tt.eq(sex, 0), 15.0116, 15.5783),
                                        tt.switch(tt.eq(sex, 0), 15.2106, 16.5573)))

    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m