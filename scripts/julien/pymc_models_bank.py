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
from mb_modelbase.utils.data_type_mapper import DataTypeMapper

filepath = os.path.join(os.path.dirname(__file__), "data", "bank_cleaned.csv")
df = pd.read_csv(filepath)

bank_backward_map = {
    'job': {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4, 'retired': 5,
            'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11},
    'marital': {'divorced': 0, 'married': 1, 'single': 2},
    'education': {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}, 'default': {'no': 0, 'yes': 1},
    'housing': {'no': 0, 'yes': 1}, 'loan': {'no': 0, 'yes': 1},
    'contact': {'cellular': 0, 'telephone': 1, 'unknown': 2},
    'month': {'apr': 0, 'aug': 1, 'dec': 2, 'feb': 3, 'jan': 4, 'jul': 5, 'jun': 6, 'mar': 7, 'may': 8, 'nov': 9,
              'oct': 10, 'sep': 11}, 'poutcome': {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3},
    'y': {'no': 0, 'yes': 1}
}

dtm = DataTypeMapper()
for name, map_ in bank_backward_map.items():
    dtm.set_map(forward='auto', backward=map_, name=name)

sample_size = 10000


