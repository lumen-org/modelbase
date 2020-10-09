#!usr/bin/python
# -*- coding: utf-8 -*-import string

from mb.modelbase import DataTypeMapper

import bin.experiments.allbus as allbus_data

# LOAD FILES
test_data = allbus_data.train(numeric_happy=False)
train_data = allbus_data.test(numeric_happy=False)

df = train_data

# SAVE PARAMETER IN THIS FILE
model_file = 'allbus_results.dat'
continues_data_file = 'allbus_happiness_values.dat'

sample_size = 15000

allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                       'lived_abroad': {'No': 0, 'Yes': 1}}

allbus_backward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                      'lived_abroad': {0: 'No', 1: 'Yes'}}

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)

