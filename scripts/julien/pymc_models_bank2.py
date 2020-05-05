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

#####################
# 437 parameter
#####################
def create_bank_tabu_biccg(filename="", modelname="bank_tabu_biccg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), [0.0953,0.2096,0.0062,0.0687,0.0445,0.1989,0.0252,0.0131,0.1827,0.1191,0.0221,0.0145], tt.switch(tt.eq(contact, 1), [0.0565,0.0664,0.0066,0.0764,0.0598,0.3389,0.0266,0.0332,0.1628,0.1262,0.0365,0.01], [0.0,0.0045,0.0,0.0,0.0008,0.0211,0.3399,0.0008,0.6193,0.0045,0.0038,0.0053])))
        housing = pm.Categorical('housing', p=tt.switch(tt.eq(month, 0), [0.2867,0.7133], tt.switch(tt.eq(month, 1), [0.8041,0.1959], tt.switch(tt.eq(month, 2), [0.7,0.3], tt.switch(tt.eq(month, 3), [0.5856,0.4144], tt.switch(tt.eq(month, 4), [0.5946,0.4054], tt.switch(tt.eq(month, 5), [0.5113,0.4887], tt.switch(tt.eq(month, 6), [0.5499,0.4501], tt.switch(tt.eq(month, 7), [0.7755,0.2245], tt.switch(tt.eq(month, 8), [0.128,0.872], tt.switch(tt.eq(month, 9), [0.4242,0.5758], tt.switch(tt.eq(month, 10), [0.8125,0.1875], [0.7115,0.2885]))))))))))))
        loan = pm.Categorical('loan', p=tt.switch(tt.eq(month, 0), [0.8908,0.1092], tt.switch(tt.eq(month, 1), [0.8942,0.1058], tt.switch(tt.eq(month, 2), [0.95,0.05], tt.switch(tt.eq(month, 3), [0.8604,0.1396], tt.switch(tt.eq(month, 4), [0.8784,0.1216], tt.switch(tt.eq(month, 5), [0.7025,0.2975], tt.switch(tt.eq(month, 6), [0.8832,0.1168], tt.switch(tt.eq(month, 7), [0.9796,0.0204], tt.switch(tt.eq(month, 8), [0.8648,0.1352], tt.switch(tt.eq(month, 9), [0.8201,0.1799], tt.switch(tt.eq(month, 10), [0.9125,0.0875], [0.9423,0.0577]))))))))))))
        campaign = pm.Normal('campaign', mu=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.8929, 1.8565), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.9273, 3.9516), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 2.1429, 1.1667), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.2231, 2.2935), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 1.8068, 1.7833), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.1745, 4.2377), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.5377, 3.9456), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.6316, 2.8182), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.3408, 2.4537), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.897, 1.9866), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 1.4923, 1.2), tt.switch(tt.eq(housing, 0), 1.4595, 2.0667)))))))))))), sigma=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.7217, 1.2163), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.5667, 3.763), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 1.6104, 0.4082), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.0993, 1.661), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 0.9451, 1.075), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.0322, 5.5281), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.3584, 5.0261), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.5619, 2.1826), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.5508, 2.413), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.3142, 1.4532), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 0.9375, 0.5606), tt.switch(tt.eq(housing, 0), 0.6053, 1.6676)))))))))))))
        poutcome = pm.Categorical('poutcome', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(contact, 0), [0.0858,0.0377,0.0544,0.8222], tt.switch(tt.eq(contact, 1), [0.1272,0.052,0.0636,0.7572], [0.0056,0.0028,0.0,0.9915])), tt.switch(tt.eq(contact, 0), [0.2216,0.08,0.0246,0.6737], tt.switch(tt.eq(contact, 1), [0.1328,0.0938,0.0156,0.7578], [0.0021,0.0041,0.0021,0.9917]))))
        y = pm.Categorical('y', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(poutcome, 0), [0.8163,0.1837], tt.switch(tt.eq(poutcome, 1), [0.7344,0.2656], tt.switch(tt.eq(poutcome, 2), [0.2921,0.7079], [0.8833,0.1167]))), tt.switch(tt.eq(poutcome, 0), [0.895,0.105], tt.switch(tt.eq(poutcome, 1), [0.8421,0.1579], tt.switch(tt.eq(poutcome, 2), [0.5,0.5], [0.93,0.07])))))
        job = pm.Categorical('job', p=tt.switch(tt.eq(housing, 0), [0.0897,0.1279,0.0377,0.0372,0.2375,0.0917,0.0484,0.0673,0.0326,0.1753,0.0357,0.0189], [0.118,0.2716,0.0367,0.0152,0.1966,0.0195,0.0344,0.1114,0.0078,0.1657,0.0227,0.0004]))
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), [0.0356,0.8222,0.1067,0.0356], tt.switch(tt.eq(job, 1), [0.3901,0.5539,0.0127,0.0433], tt.switch(tt.eq(job, 2), [0.1548,0.3452,0.4345,0.0655], tt.switch(tt.eq(job, 3), [0.5089,0.25,0.1964,0.0446], tt.switch(tt.eq(job, 4), [0.0402,0.1197,0.8122,0.0279], tt.switch(tt.eq(job, 5), [0.3478,0.4565,0.1348,0.0609], tt.switch(tt.eq(job, 6), [0.082,0.4153,0.4809,0.0219], tt.switch(tt.eq(job, 7), [0.06,0.8705,0.0384,0.0312], tt.switch(tt.eq(job, 8), [0.0238,0.5595,0.2262,0.1905], tt.switch(tt.eq(job, 9), [0.0195,0.6771,0.2747,0.0286], tt.switch(tt.eq(job, 10), [0.2031,0.5312,0.25,0.0156], [0.1842,0.2105,0.2105,0.3947]))))))))))))
        default = pm.Categorical('default', p=tt.switch(tt.eq(loan, 0), [0.9867,0.0133], [0.9638,0.0362]))
        balance = pm.Normal('balance', mu=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 1609.7468, tt.switch(tt.eq(month, 1), 1430.222, tt.switch(tt.eq(month, 2), 5108.1818, tt.switch(tt.eq(month, 3), 1199.2174, tt.switch(tt.eq(month, 4), 826.0, tt.switch(tt.eq(month, 5), 783.3798, tt.switch(tt.eq(month, 6), 1917.8004, tt.switch(tt.eq(month, 7), 2120.6071, tt.switch(tt.eq(month, 8), 1096.7778, tt.switch(tt.eq(month, 9), 2649.4514, tt.switch(tt.eq(month, 10), 4045.6512, 1229.1429))))))))))), tt.switch(tt.eq(month, 0), 1862.5357, tt.switch(tt.eq(month, 1), 1625.2278, tt.switch(tt.eq(month, 2), 1683.6667, tt.switch(tt.eq(month, 3), 1899.2368, tt.switch(tt.eq(month, 4), 2216.875, tt.switch(tt.eq(month, 5), 853.8361, tt.switch(tt.eq(month, 6), 1599.1273, tt.switch(tt.eq(month, 7), 1954.7619, tt.switch(tt.eq(month, 8), 1207.957, tt.switch(tt.eq(month, 9), 2187.6667, tt.switch(tt.eq(month, 10), 1219.7027, 2330.9412)))))))))))), sigma=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 2842.5471, tt.switch(tt.eq(month, 1), 3202.5655, tt.switch(tt.eq(month, 2), 5501.1073, tt.switch(tt.eq(month, 3), 2478.4931, tt.switch(tt.eq(month, 4), 1813.7531, tt.switch(tt.eq(month, 5), 2018.1316, tt.switch(tt.eq(month, 6), 3739.245, tt.switch(tt.eq(month, 7), 2565.8502, tt.switch(tt.eq(month, 8), 2375.6243, tt.switch(tt.eq(month, 9), 3644.7028, tt.switch(tt.eq(month, 10), 11222.2315, 1622.7708))))))))))), tt.switch(tt.eq(month, 0), 3811.0455, tt.switch(tt.eq(month, 1), 2786.9022, tt.switch(tt.eq(month, 2), 1382.0483, tt.switch(tt.eq(month, 3), 2055.7174, tt.switch(tt.eq(month, 4), 2284.5375, tt.switch(tt.eq(month, 5), 1543.8069, tt.switch(tt.eq(month, 6), 2037.8572, tt.switch(tt.eq(month, 7), 2670.6014, tt.switch(tt.eq(month, 8), 1890.1508, tt.switch(tt.eq(month, 9), 1970.479, tt.switch(tt.eq(month, 10), 1542.2663, 4249.0182)))))))))))))
        day = pm.Normal('day', mu=tt.switch(tt.eq(month, 0), campaign*-0.0894+17.6413, tt.switch(tt.eq(month, 1), campaign*0.8894+12.9831, tt.switch(tt.eq(month, 2), campaign*1.1414+13.8885, tt.switch(tt.eq(month, 3), campaign*0.1137+6.2935, tt.switch(tt.eq(month, 4), campaign*0.5229+26.8035, tt.switch(tt.eq(month, 5), campaign*0.58+16.1692, tt.switch(tt.eq(month, 6), campaign*0.1095+10.6508, tt.switch(tt.eq(month, 7), campaign*0.1055+13.6976, tt.switch(tt.eq(month, 8), campaign*0.3315+14.7494, tt.switch(tt.eq(month, 9), campaign*0.3216+17.8643, tt.switch(tt.eq(month, 10), campaign*1.1905+16.6762, campaign*-1.4991+15.0274))))))))))), sigma=tt.switch(tt.eq(month, 0), 6.6706, tt.switch(tt.eq(month, 1), 7.2523, tt.switch(tt.eq(month, 2), 9.1159, tt.switch(tt.eq(month, 3), 5.873, tt.switch(tt.eq(month, 4), 4.0242, tt.switch(tt.eq(month, 5), 8.2104, tt.switch(tt.eq(month, 6), 7.5395, tt.switch(tt.eq(month, 7), 9.0996, tt.switch(tt.eq(month, 8), 7.6423, tt.switch(tt.eq(month, 9), 3.2162, tt.switch(tt.eq(month, 10), 7.6339, 8.691))))))))))))
        duration = pm.Normal('duration', mu=tt.switch(tt.eq(y, 0), campaign*-6.2918+244.3562, campaign*43.7177+453.6438), sigma=tt.switch(tt.eq(y, 0), 209.3662, 379.8245))
        pdays = pm.Normal('pdays', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), day*-1.8824+88.705, day*-0.6779+56.3173), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), day*-2.729+100.9352, day*-0.6742+24.0741), tt.switch(tt.eq(loan, 0), day*-0.0036+0.4453, day*-1.1641+24.7693))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), 115.8252, 99.7195), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), 103.3666, 49.2519), tt.switch(tt.eq(loan, 0), 20.5746, 75.2849))))
        previous = pm.Normal('previous', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), pdays*0.0116+0.1781, pdays*0.0088+0.2565), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), pdays*0.0114+0.3003, pdays*0.0087+0.2251), tt.switch(tt.eq(housing, 0), pdays*0.0155+0.0278, pdays*0.0063+0.0145))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), 1.3013, 1.9096), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), 1.6119, 1.4071), tt.switch(tt.eq(housing, 0), 0.2081, 0.2048))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(job, 0), 39.682, tt.switch(tt.eq(job, 1), 40.1564, tt.switch(tt.eq(job, 2), 42.0119, tt.switch(tt.eq(job, 3), 47.3393, tt.switch(tt.eq(job, 4), 40.5408, tt.switch(tt.eq(job, 5), 61.8696, tt.switch(tt.eq(job, 6), 41.4536, tt.switch(tt.eq(job, 7), 38.5707, tt.switch(tt.eq(job, 8), 26.8214, tt.switch(tt.eq(job, 9), 39.4701, tt.switch(tt.eq(job, 10), 40.9062, 48.1053))))))))))), sigma=tt.switch(tt.eq(job, 0), 9.4425, tt.switch(tt.eq(job, 1), 9.0381, tt.switch(tt.eq(job, 2), 8.358, tt.switch(tt.eq(job, 3), 11.0082, tt.switch(tt.eq(job, 4), 9.1924, tt.switch(tt.eq(job, 5), 9.7895, tt.switch(tt.eq(job, 6), 9.407, tt.switch(tt.eq(job, 7), 9.2078, tt.switch(tt.eq(job, 8), 5.2828, tt.switch(tt.eq(job, 9), 8.6716, tt.switch(tt.eq(job, 10), 9.7653, 10.3996))))))))))))
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 437 parameter
#####################
def create_bank_tabu_predloglikcg(filename="", modelname="bank_tabu_predloglikcg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), [0.0953,0.2096,0.0062,0.0687,0.0445,0.1989,0.0252,0.0131,0.1827,0.1191,0.0221,0.0145], tt.switch(tt.eq(contact, 1), [0.0565,0.0664,0.0066,0.0764,0.0598,0.3389,0.0266,0.0332,0.1628,0.1262,0.0365,0.01], [0.0,0.0045,0.0,0.0,0.0008,0.0211,0.3399,0.0008,0.6193,0.0045,0.0038,0.0053])))
        housing = pm.Categorical('housing', p=tt.switch(tt.eq(month, 0), [0.2867,0.7133], tt.switch(tt.eq(month, 1), [0.8041,0.1959], tt.switch(tt.eq(month, 2), [0.7,0.3], tt.switch(tt.eq(month, 3), [0.5856,0.4144], tt.switch(tt.eq(month, 4), [0.5946,0.4054], tt.switch(tt.eq(month, 5), [0.5113,0.4887], tt.switch(tt.eq(month, 6), [0.5499,0.4501], tt.switch(tt.eq(month, 7), [0.7755,0.2245], tt.switch(tt.eq(month, 8), [0.128,0.872], tt.switch(tt.eq(month, 9), [0.4242,0.5758], tt.switch(tt.eq(month, 10), [0.8125,0.1875], [0.7115,0.2885]))))))))))))
        loan = pm.Categorical('loan', p=tt.switch(tt.eq(month, 0), [0.8908,0.1092], tt.switch(tt.eq(month, 1), [0.8942,0.1058], tt.switch(tt.eq(month, 2), [0.95,0.05], tt.switch(tt.eq(month, 3), [0.8604,0.1396], tt.switch(tt.eq(month, 4), [0.8784,0.1216], tt.switch(tt.eq(month, 5), [0.7025,0.2975], tt.switch(tt.eq(month, 6), [0.8832,0.1168], tt.switch(tt.eq(month, 7), [0.9796,0.0204], tt.switch(tt.eq(month, 8), [0.8648,0.1352], tt.switch(tt.eq(month, 9), [0.8201,0.1799], tt.switch(tt.eq(month, 10), [0.9125,0.0875], [0.9423,0.0577]))))))))))))
        campaign = pm.Normal('campaign', mu=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.8929, 1.8565), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.9273, 3.9516), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 2.1429, 1.1667), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.2231, 2.2935), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 1.8068, 1.7833), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.1745, 4.2377), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.5377, 3.9456), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.6316, 2.8182), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.3408, 2.4537), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.897, 1.9866), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 1.4923, 1.2), tt.switch(tt.eq(housing, 0), 1.4595, 2.0667)))))))))))), sigma=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.7217, 1.2163), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.5667, 3.763), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 1.6104, 0.4082), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.0993, 1.661), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 0.9451, 1.075), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.0322, 5.5281), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.3584, 5.0261), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.5619, 2.1826), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.5508, 2.413), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.3142, 1.4532), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 0.9375, 0.5606), tt.switch(tt.eq(housing, 0), 0.6053, 1.6676)))))))))))))
        poutcome = pm.Categorical('poutcome', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(contact, 0), [0.0858,0.0377,0.0544,0.8222], tt.switch(tt.eq(contact, 1), [0.1272,0.052,0.0636,0.7572], [0.0056,0.0028,0.0,0.9915])), tt.switch(tt.eq(contact, 0), [0.2216,0.08,0.0246,0.6737], tt.switch(tt.eq(contact, 1), [0.1328,0.0938,0.0156,0.7578], [0.0021,0.0041,0.0021,0.9917]))))
        y = pm.Categorical('y', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(poutcome, 0), [0.8163,0.1837], tt.switch(tt.eq(poutcome, 1), [0.7344,0.2656], tt.switch(tt.eq(poutcome, 2), [0.2921,0.7079], [0.8833,0.1167]))), tt.switch(tt.eq(poutcome, 0), [0.895,0.105], tt.switch(tt.eq(poutcome, 1), [0.8421,0.1579], tt.switch(tt.eq(poutcome, 2), [0.5,0.5], [0.93,0.07])))))
        job = pm.Categorical('job', p=tt.switch(tt.eq(housing, 0), [0.0897,0.1279,0.0377,0.0372,0.2375,0.0917,0.0484,0.0673,0.0326,0.1753,0.0357,0.0189], [0.118,0.2716,0.0367,0.0152,0.1966,0.0195,0.0344,0.1114,0.0078,0.1657,0.0227,0.0004]))
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), [0.0356,0.8222,0.1067,0.0356], tt.switch(tt.eq(job, 1), [0.3901,0.5539,0.0127,0.0433], tt.switch(tt.eq(job, 2), [0.1548,0.3452,0.4345,0.0655], tt.switch(tt.eq(job, 3), [0.5089,0.25,0.1964,0.0446], tt.switch(tt.eq(job, 4), [0.0402,0.1197,0.8122,0.0279], tt.switch(tt.eq(job, 5), [0.3478,0.4565,0.1348,0.0609], tt.switch(tt.eq(job, 6), [0.082,0.4153,0.4809,0.0219], tt.switch(tt.eq(job, 7), [0.06,0.8705,0.0384,0.0312], tt.switch(tt.eq(job, 8), [0.0238,0.5595,0.2262,0.1905], tt.switch(tt.eq(job, 9), [0.0195,0.6771,0.2747,0.0286], tt.switch(tt.eq(job, 10), [0.2031,0.5312,0.25,0.0156], [0.1842,0.2105,0.2105,0.3947]))))))))))))
        default = pm.Categorical('default', p=tt.switch(tt.eq(loan, 0), [0.9867,0.0133], [0.9638,0.0362]))
        balance = pm.Normal('balance', mu=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 1609.7468, tt.switch(tt.eq(month, 1), 1430.222, tt.switch(tt.eq(month, 2), 5108.1818, tt.switch(tt.eq(month, 3), 1199.2174, tt.switch(tt.eq(month, 4), 826.0, tt.switch(tt.eq(month, 5), 783.3798, tt.switch(tt.eq(month, 6), 1917.8004, tt.switch(tt.eq(month, 7), 2120.6071, tt.switch(tt.eq(month, 8), 1096.7778, tt.switch(tt.eq(month, 9), 2649.4514, tt.switch(tt.eq(month, 10), 4045.6512, 1229.1429))))))))))), tt.switch(tt.eq(month, 0), 1862.5357, tt.switch(tt.eq(month, 1), 1625.2278, tt.switch(tt.eq(month, 2), 1683.6667, tt.switch(tt.eq(month, 3), 1899.2368, tt.switch(tt.eq(month, 4), 2216.875, tt.switch(tt.eq(month, 5), 853.8361, tt.switch(tt.eq(month, 6), 1599.1273, tt.switch(tt.eq(month, 7), 1954.7619, tt.switch(tt.eq(month, 8), 1207.957, tt.switch(tt.eq(month, 9), 2187.6667, tt.switch(tt.eq(month, 10), 1219.7027, 2330.9412)))))))))))), sigma=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 2842.5471, tt.switch(tt.eq(month, 1), 3202.5655, tt.switch(tt.eq(month, 2), 5501.1073, tt.switch(tt.eq(month, 3), 2478.4931, tt.switch(tt.eq(month, 4), 1813.7531, tt.switch(tt.eq(month, 5), 2018.1316, tt.switch(tt.eq(month, 6), 3739.245, tt.switch(tt.eq(month, 7), 2565.8502, tt.switch(tt.eq(month, 8), 2375.6243, tt.switch(tt.eq(month, 9), 3644.7028, tt.switch(tt.eq(month, 10), 11222.2315, 1622.7708))))))))))), tt.switch(tt.eq(month, 0), 3811.0455, tt.switch(tt.eq(month, 1), 2786.9022, tt.switch(tt.eq(month, 2), 1382.0483, tt.switch(tt.eq(month, 3), 2055.7174, tt.switch(tt.eq(month, 4), 2284.5375, tt.switch(tt.eq(month, 5), 1543.8069, tt.switch(tt.eq(month, 6), 2037.8572, tt.switch(tt.eq(month, 7), 2670.6014, tt.switch(tt.eq(month, 8), 1890.1508, tt.switch(tt.eq(month, 9), 1970.479, tt.switch(tt.eq(month, 10), 1542.2663, 4249.0182)))))))))))))
        day = pm.Normal('day', mu=tt.switch(tt.eq(month, 0), campaign*-0.0894+17.6413, tt.switch(tt.eq(month, 1), campaign*0.8894+12.9831, tt.switch(tt.eq(month, 2), campaign*1.1414+13.8885, tt.switch(tt.eq(month, 3), campaign*0.1137+6.2935, tt.switch(tt.eq(month, 4), campaign*0.5229+26.8035, tt.switch(tt.eq(month, 5), campaign*0.58+16.1692, tt.switch(tt.eq(month, 6), campaign*0.1095+10.6508, tt.switch(tt.eq(month, 7), campaign*0.1055+13.6976, tt.switch(tt.eq(month, 8), campaign*0.3315+14.7494, tt.switch(tt.eq(month, 9), campaign*0.3216+17.8643, tt.switch(tt.eq(month, 10), campaign*1.1905+16.6762, campaign*-1.4991+15.0274))))))))))), sigma=tt.switch(tt.eq(month, 0), 6.6706, tt.switch(tt.eq(month, 1), 7.2523, tt.switch(tt.eq(month, 2), 9.1159, tt.switch(tt.eq(month, 3), 5.873, tt.switch(tt.eq(month, 4), 4.0242, tt.switch(tt.eq(month, 5), 8.2104, tt.switch(tt.eq(month, 6), 7.5395, tt.switch(tt.eq(month, 7), 9.0996, tt.switch(tt.eq(month, 8), 7.6423, tt.switch(tt.eq(month, 9), 3.2162, tt.switch(tt.eq(month, 10), 7.6339, 8.691))))))))))))
        duration = pm.Normal('duration', mu=tt.switch(tt.eq(y, 0), campaign*-6.2918+244.3562, campaign*43.7177+453.6438), sigma=tt.switch(tt.eq(y, 0), 209.3662, 379.8245))
        pdays = pm.Normal('pdays', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), day*-1.8824+88.705, day*-0.6779+56.3173), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), day*-2.729+100.9352, day*-0.6742+24.0741), tt.switch(tt.eq(loan, 0), day*-0.0036+0.4453, day*-1.1641+24.7693))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), 115.8252, 99.7195), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), 103.3666, 49.2519), tt.switch(tt.eq(loan, 0), 20.5746, 75.2849))))
        previous = pm.Normal('previous', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), pdays*0.0116+0.1781, pdays*0.0088+0.2565), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), pdays*0.0114+0.3003, pdays*0.0087+0.2251), tt.switch(tt.eq(housing, 0), pdays*0.0155+0.0278, pdays*0.0063+0.0145))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), 1.3013, 1.9096), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), 1.6119, 1.4071), tt.switch(tt.eq(housing, 0), 0.2081, 0.2048))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(job, 0), 39.682, tt.switch(tt.eq(job, 1), 40.1564, tt.switch(tt.eq(job, 2), 42.0119, tt.switch(tt.eq(job, 3), 47.3393, tt.switch(tt.eq(job, 4), 40.5408, tt.switch(tt.eq(job, 5), 61.8696, tt.switch(tt.eq(job, 6), 41.4536, tt.switch(tt.eq(job, 7), 38.5707, tt.switch(tt.eq(job, 8), 26.8214, tt.switch(tt.eq(job, 9), 39.4701, tt.switch(tt.eq(job, 10), 40.9062, 48.1053))))))))))), sigma=tt.switch(tt.eq(job, 0), 9.4425, tt.switch(tt.eq(job, 1), 9.0381, tt.switch(tt.eq(job, 2), 8.358, tt.switch(tt.eq(job, 3), 11.0082, tt.switch(tt.eq(job, 4), 9.1924, tt.switch(tt.eq(job, 5), 9.7895, tt.switch(tt.eq(job, 6), 9.407, tt.switch(tt.eq(job, 7), 9.2078, tt.switch(tt.eq(job, 8), 5.2828, tt.switch(tt.eq(job, 9), 8.6716, tt.switch(tt.eq(job, 10), 9.7653, 10.3996))))))))))))
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 437 parameter
#####################
def create_bank_hc_biccg(filename="", modelname="bank_hc_biccg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), [0.0953,0.2096,0.0062,0.0687,0.0445,0.1989,0.0252,0.0131,0.1827,0.1191,0.0221,0.0145], tt.switch(tt.eq(contact, 1), [0.0565,0.0664,0.0066,0.0764,0.0598,0.3389,0.0266,0.0332,0.1628,0.1262,0.0365,0.01], [0.0,0.0045,0.0,0.0,0.0008,0.0211,0.3399,0.0008,0.6193,0.0045,0.0038,0.0053])))
        housing = pm.Categorical('housing', p=tt.switch(tt.eq(month, 0), [0.2867,0.7133], tt.switch(tt.eq(month, 1), [0.8041,0.1959], tt.switch(tt.eq(month, 2), [0.7,0.3], tt.switch(tt.eq(month, 3), [0.5856,0.4144], tt.switch(tt.eq(month, 4), [0.5946,0.4054], tt.switch(tt.eq(month, 5), [0.5113,0.4887], tt.switch(tt.eq(month, 6), [0.5499,0.4501], tt.switch(tt.eq(month, 7), [0.7755,0.2245], tt.switch(tt.eq(month, 8), [0.128,0.872], tt.switch(tt.eq(month, 9), [0.4242,0.5758], tt.switch(tt.eq(month, 10), [0.8125,0.1875], [0.7115,0.2885]))))))))))))
        loan = pm.Categorical('loan', p=tt.switch(tt.eq(month, 0), [0.8908,0.1092], tt.switch(tt.eq(month, 1), [0.8942,0.1058], tt.switch(tt.eq(month, 2), [0.95,0.05], tt.switch(tt.eq(month, 3), [0.8604,0.1396], tt.switch(tt.eq(month, 4), [0.8784,0.1216], tt.switch(tt.eq(month, 5), [0.7025,0.2975], tt.switch(tt.eq(month, 6), [0.8832,0.1168], tt.switch(tt.eq(month, 7), [0.9796,0.0204], tt.switch(tt.eq(month, 8), [0.8648,0.1352], tt.switch(tt.eq(month, 9), [0.8201,0.1799], tt.switch(tt.eq(month, 10), [0.9125,0.0875], [0.9423,0.0577]))))))))))))
        campaign = pm.Normal('campaign', mu=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.8929, 1.8565), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.9273, 3.9516), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 2.1429, 1.1667), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.2231, 2.2935), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 1.8068, 1.7833), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.1745, 4.2377), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.5377, 3.9456), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.6316, 2.8182), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.3408, 2.4537), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.897, 1.9866), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 1.4923, 1.2), tt.switch(tt.eq(housing, 0), 1.4595, 2.0667)))))))))))), sigma=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.7217, 1.2163), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.5667, 3.763), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 1.6104, 0.4082), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.0993, 1.661), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 0.9451, 1.075), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.0322, 5.5281), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.3584, 5.0261), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.5619, 2.1826), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.5508, 2.413), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.3142, 1.4532), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 0.9375, 0.5606), tt.switch(tt.eq(housing, 0), 0.6053, 1.6676)))))))))))))
        poutcome = pm.Categorical('poutcome', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(contact, 0), [0.0858,0.0377,0.0544,0.8222], tt.switch(tt.eq(contact, 1), [0.1272,0.052,0.0636,0.7572], [0.0056,0.0028,0.0,0.9915])), tt.switch(tt.eq(contact, 0), [0.2216,0.08,0.0246,0.6737], tt.switch(tt.eq(contact, 1), [0.1328,0.0938,0.0156,0.7578], [0.0021,0.0041,0.0021,0.9917]))))
        y = pm.Categorical('y', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(poutcome, 0), [0.8163,0.1837], tt.switch(tt.eq(poutcome, 1), [0.7344,0.2656], tt.switch(tt.eq(poutcome, 2), [0.2921,0.7079], [0.8833,0.1167]))), tt.switch(tt.eq(poutcome, 0), [0.895,0.105], tt.switch(tt.eq(poutcome, 1), [0.8421,0.1579], tt.switch(tt.eq(poutcome, 2), [0.5,0.5], [0.93,0.07])))))
        job = pm.Categorical('job', p=tt.switch(tt.eq(housing, 0), [0.0897,0.1279,0.0377,0.0372,0.2375,0.0917,0.0484,0.0673,0.0326,0.1753,0.0357,0.0189], [0.118,0.2716,0.0367,0.0152,0.1966,0.0195,0.0344,0.1114,0.0078,0.1657,0.0227,0.0004]))
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), [0.0356,0.8222,0.1067,0.0356], tt.switch(tt.eq(job, 1), [0.3901,0.5539,0.0127,0.0433], tt.switch(tt.eq(job, 2), [0.1548,0.3452,0.4345,0.0655], tt.switch(tt.eq(job, 3), [0.5089,0.25,0.1964,0.0446], tt.switch(tt.eq(job, 4), [0.0402,0.1197,0.8122,0.0279], tt.switch(tt.eq(job, 5), [0.3478,0.4565,0.1348,0.0609], tt.switch(tt.eq(job, 6), [0.082,0.4153,0.4809,0.0219], tt.switch(tt.eq(job, 7), [0.06,0.8705,0.0384,0.0312], tt.switch(tt.eq(job, 8), [0.0238,0.5595,0.2262,0.1905], tt.switch(tt.eq(job, 9), [0.0195,0.6771,0.2747,0.0286], tt.switch(tt.eq(job, 10), [0.2031,0.5312,0.25,0.0156], [0.1842,0.2105,0.2105,0.3947]))))))))))))
        default = pm.Categorical('default', p=tt.switch(tt.eq(loan, 0), [0.9867,0.0133], [0.9638,0.0362]))
        balance = pm.Normal('balance', mu=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 1609.7468, tt.switch(tt.eq(month, 1), 1430.222, tt.switch(tt.eq(month, 2), 5108.1818, tt.switch(tt.eq(month, 3), 1199.2174, tt.switch(tt.eq(month, 4), 826.0, tt.switch(tt.eq(month, 5), 783.3798, tt.switch(tt.eq(month, 6), 1917.8004, tt.switch(tt.eq(month, 7), 2120.6071, tt.switch(tt.eq(month, 8), 1096.7778, tt.switch(tt.eq(month, 9), 2649.4514, tt.switch(tt.eq(month, 10), 4045.6512, 1229.1429))))))))))), tt.switch(tt.eq(month, 0), 1862.5357, tt.switch(tt.eq(month, 1), 1625.2278, tt.switch(tt.eq(month, 2), 1683.6667, tt.switch(tt.eq(month, 3), 1899.2368, tt.switch(tt.eq(month, 4), 2216.875, tt.switch(tt.eq(month, 5), 853.8361, tt.switch(tt.eq(month, 6), 1599.1273, tt.switch(tt.eq(month, 7), 1954.7619, tt.switch(tt.eq(month, 8), 1207.957, tt.switch(tt.eq(month, 9), 2187.6667, tt.switch(tt.eq(month, 10), 1219.7027, 2330.9412)))))))))))), sigma=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 2842.5471, tt.switch(tt.eq(month, 1), 3202.5655, tt.switch(tt.eq(month, 2), 5501.1073, tt.switch(tt.eq(month, 3), 2478.4931, tt.switch(tt.eq(month, 4), 1813.7531, tt.switch(tt.eq(month, 5), 2018.1316, tt.switch(tt.eq(month, 6), 3739.245, tt.switch(tt.eq(month, 7), 2565.8502, tt.switch(tt.eq(month, 8), 2375.6243, tt.switch(tt.eq(month, 9), 3644.7028, tt.switch(tt.eq(month, 10), 11222.2315, 1622.7708))))))))))), tt.switch(tt.eq(month, 0), 3811.0455, tt.switch(tt.eq(month, 1), 2786.9022, tt.switch(tt.eq(month, 2), 1382.0483, tt.switch(tt.eq(month, 3), 2055.7174, tt.switch(tt.eq(month, 4), 2284.5375, tt.switch(tt.eq(month, 5), 1543.8069, tt.switch(tt.eq(month, 6), 2037.8572, tt.switch(tt.eq(month, 7), 2670.6014, tt.switch(tt.eq(month, 8), 1890.1508, tt.switch(tt.eq(month, 9), 1970.479, tt.switch(tt.eq(month, 10), 1542.2663, 4249.0182)))))))))))))
        day = pm.Normal('day', mu=tt.switch(tt.eq(month, 0), campaign*-0.0894+17.6413, tt.switch(tt.eq(month, 1), campaign*0.8894+12.9831, tt.switch(tt.eq(month, 2), campaign*1.1414+13.8885, tt.switch(tt.eq(month, 3), campaign*0.1137+6.2935, tt.switch(tt.eq(month, 4), campaign*0.5229+26.8035, tt.switch(tt.eq(month, 5), campaign*0.58+16.1692, tt.switch(tt.eq(month, 6), campaign*0.1095+10.6508, tt.switch(tt.eq(month, 7), campaign*0.1055+13.6976, tt.switch(tt.eq(month, 8), campaign*0.3315+14.7494, tt.switch(tt.eq(month, 9), campaign*0.3216+17.8643, tt.switch(tt.eq(month, 10), campaign*1.1905+16.6762, campaign*-1.4991+15.0274))))))))))), sigma=tt.switch(tt.eq(month, 0), 6.6706, tt.switch(tt.eq(month, 1), 7.2523, tt.switch(tt.eq(month, 2), 9.1159, tt.switch(tt.eq(month, 3), 5.873, tt.switch(tt.eq(month, 4), 4.0242, tt.switch(tt.eq(month, 5), 8.2104, tt.switch(tt.eq(month, 6), 7.5395, tt.switch(tt.eq(month, 7), 9.0996, tt.switch(tt.eq(month, 8), 7.6423, tt.switch(tt.eq(month, 9), 3.2162, tt.switch(tt.eq(month, 10), 7.6339, 8.691))))))))))))
        duration = pm.Normal('duration', mu=tt.switch(tt.eq(y, 0), campaign*-6.2918+244.3562, campaign*43.7177+453.6438), sigma=tt.switch(tt.eq(y, 0), 209.3662, 379.8245))
        pdays = pm.Normal('pdays', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), day*-1.8824+88.705, day*-0.6779+56.3173), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), day*-2.729+100.9352, day*-0.6742+24.0741), tt.switch(tt.eq(loan, 0), day*-0.0036+0.4453, day*-1.1641+24.7693))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), 115.8252, 99.7195), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), 103.3666, 49.2519), tt.switch(tt.eq(loan, 0), 20.5746, 75.2849))))
        previous = pm.Normal('previous', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), pdays*0.0116+0.1781, pdays*0.0088+0.2565), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), pdays*0.0114+0.3003, pdays*0.0087+0.2251), tt.switch(tt.eq(housing, 0), pdays*0.0155+0.0278, pdays*0.0063+0.0145))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), 1.3013, 1.9096), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), 1.6119, 1.4071), tt.switch(tt.eq(housing, 0), 0.2081, 0.2048))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(job, 0), 39.682, tt.switch(tt.eq(job, 1), 40.1564, tt.switch(tt.eq(job, 2), 42.0119, tt.switch(tt.eq(job, 3), 47.3393, tt.switch(tt.eq(job, 4), 40.5408, tt.switch(tt.eq(job, 5), 61.8696, tt.switch(tt.eq(job, 6), 41.4536, tt.switch(tt.eq(job, 7), 38.5707, tt.switch(tt.eq(job, 8), 26.8214, tt.switch(tt.eq(job, 9), 39.4701, tt.switch(tt.eq(job, 10), 40.9062, 48.1053))))))))))), sigma=tt.switch(tt.eq(job, 0), 9.4425, tt.switch(tt.eq(job, 1), 9.0381, tt.switch(tt.eq(job, 2), 8.358, tt.switch(tt.eq(job, 3), 11.0082, tt.switch(tt.eq(job, 4), 9.1924, tt.switch(tt.eq(job, 5), 9.7895, tt.switch(tt.eq(job, 6), 9.407, tt.switch(tt.eq(job, 7), 9.2078, tt.switch(tt.eq(job, 8), 5.2828, tt.switch(tt.eq(job, 9), 8.6716, tt.switch(tt.eq(job, 10), 9.7653, 10.3996))))))))))))
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 437 parameter
#####################
def create_bank_hc_predloglikcg(filename="", modelname="bank_hc_predloglikcg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), [0.0953,0.2096,0.0062,0.0687,0.0445,0.1989,0.0252,0.0131,0.1827,0.1191,0.0221,0.0145], tt.switch(tt.eq(contact, 1), [0.0565,0.0664,0.0066,0.0764,0.0598,0.3389,0.0266,0.0332,0.1628,0.1262,0.0365,0.01], [0.0,0.0045,0.0,0.0,0.0008,0.0211,0.3399,0.0008,0.6193,0.0045,0.0038,0.0053])))
        housing = pm.Categorical('housing', p=tt.switch(tt.eq(month, 0), [0.2867,0.7133], tt.switch(tt.eq(month, 1), [0.8041,0.1959], tt.switch(tt.eq(month, 2), [0.7,0.3], tt.switch(tt.eq(month, 3), [0.5856,0.4144], tt.switch(tt.eq(month, 4), [0.5946,0.4054], tt.switch(tt.eq(month, 5), [0.5113,0.4887], tt.switch(tt.eq(month, 6), [0.5499,0.4501], tt.switch(tt.eq(month, 7), [0.7755,0.2245], tt.switch(tt.eq(month, 8), [0.128,0.872], tt.switch(tt.eq(month, 9), [0.4242,0.5758], tt.switch(tt.eq(month, 10), [0.8125,0.1875], [0.7115,0.2885]))))))))))))
        loan = pm.Categorical('loan', p=tt.switch(tt.eq(month, 0), [0.8908,0.1092], tt.switch(tt.eq(month, 1), [0.8942,0.1058], tt.switch(tt.eq(month, 2), [0.95,0.05], tt.switch(tt.eq(month, 3), [0.8604,0.1396], tt.switch(tt.eq(month, 4), [0.8784,0.1216], tt.switch(tt.eq(month, 5), [0.7025,0.2975], tt.switch(tt.eq(month, 6), [0.8832,0.1168], tt.switch(tt.eq(month, 7), [0.9796,0.0204], tt.switch(tt.eq(month, 8), [0.8648,0.1352], tt.switch(tt.eq(month, 9), [0.8201,0.1799], tt.switch(tt.eq(month, 10), [0.9125,0.0875], [0.9423,0.0577]))))))))))))
        campaign = pm.Normal('campaign', mu=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.8929, 1.8565), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.9273, 3.9516), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 2.1429, 1.1667), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.2231, 2.2935), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 1.8068, 1.7833), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.1745, 4.2377), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.5377, 3.9456), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.6316, 2.8182), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.3408, 2.4537), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.897, 1.9866), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 1.4923, 1.2), tt.switch(tt.eq(housing, 0), 1.4595, 2.0667)))))))))))), sigma=tt.switch(tt.eq(month, 0), tt.switch(tt.eq(housing, 0), 1.7217, 1.2163), tt.switch(tt.eq(month, 1), tt.switch(tt.eq(housing, 0), 3.5667, 3.763), tt.switch(tt.eq(month, 2), tt.switch(tt.eq(housing, 0), 1.6104, 0.4082), tt.switch(tt.eq(month, 3), tt.switch(tt.eq(housing, 0), 2.0993, 1.661), tt.switch(tt.eq(month, 4), tt.switch(tt.eq(housing, 0), 0.9451, 1.075), tt.switch(tt.eq(month, 5), tt.switch(tt.eq(housing, 0), 3.0322, 5.5281), tt.switch(tt.eq(month, 6), tt.switch(tt.eq(housing, 0), 2.3584, 5.0261), tt.switch(tt.eq(month, 7), tt.switch(tt.eq(housing, 0), 2.5619, 2.1826), tt.switch(tt.eq(month, 8), tt.switch(tt.eq(housing, 0), 2.5508, 2.413), tt.switch(tt.eq(month, 9), tt.switch(tt.eq(housing, 0), 1.3142, 1.4532), tt.switch(tt.eq(month, 10), tt.switch(tt.eq(housing, 0), 0.9375, 0.5606), tt.switch(tt.eq(housing, 0), 0.6053, 1.6676)))))))))))))
        poutcome = pm.Categorical('poutcome', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(contact, 0), [0.0858,0.0377,0.0544,0.8222], tt.switch(tt.eq(contact, 1), [0.1272,0.052,0.0636,0.7572], [0.0056,0.0028,0.0,0.9915])), tt.switch(tt.eq(contact, 0), [0.2216,0.08,0.0246,0.6737], tt.switch(tt.eq(contact, 1), [0.1328,0.0938,0.0156,0.7578], [0.0021,0.0041,0.0021,0.9917]))))
        y = pm.Categorical('y', p=tt.switch(tt.eq(housing, 0), tt.switch(tt.eq(poutcome, 0), [0.8163,0.1837], tt.switch(tt.eq(poutcome, 1), [0.7344,0.2656], tt.switch(tt.eq(poutcome, 2), [0.2921,0.7079], [0.8833,0.1167]))), tt.switch(tt.eq(poutcome, 0), [0.895,0.105], tt.switch(tt.eq(poutcome, 1), [0.8421,0.1579], tt.switch(tt.eq(poutcome, 2), [0.5,0.5], [0.93,0.07])))))
        job = pm.Categorical('job', p=tt.switch(tt.eq(housing, 0), [0.0897,0.1279,0.0377,0.0372,0.2375,0.0917,0.0484,0.0673,0.0326,0.1753,0.0357,0.0189], [0.118,0.2716,0.0367,0.0152,0.1966,0.0195,0.0344,0.1114,0.0078,0.1657,0.0227,0.0004]))
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), [0.0356,0.8222,0.1067,0.0356], tt.switch(tt.eq(job, 1), [0.3901,0.5539,0.0127,0.0433], tt.switch(tt.eq(job, 2), [0.1548,0.3452,0.4345,0.0655], tt.switch(tt.eq(job, 3), [0.5089,0.25,0.1964,0.0446], tt.switch(tt.eq(job, 4), [0.0402,0.1197,0.8122,0.0279], tt.switch(tt.eq(job, 5), [0.3478,0.4565,0.1348,0.0609], tt.switch(tt.eq(job, 6), [0.082,0.4153,0.4809,0.0219], tt.switch(tt.eq(job, 7), [0.06,0.8705,0.0384,0.0312], tt.switch(tt.eq(job, 8), [0.0238,0.5595,0.2262,0.1905], tt.switch(tt.eq(job, 9), [0.0195,0.6771,0.2747,0.0286], tt.switch(tt.eq(job, 10), [0.2031,0.5312,0.25,0.0156], [0.1842,0.2105,0.2105,0.3947]))))))))))))
        default = pm.Categorical('default', p=tt.switch(tt.eq(loan, 0), [0.9867,0.0133], [0.9638,0.0362]))
        balance = pm.Normal('balance', mu=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 1609.7468, tt.switch(tt.eq(month, 1), 1430.222, tt.switch(tt.eq(month, 2), 5108.1818, tt.switch(tt.eq(month, 3), 1199.2174, tt.switch(tt.eq(month, 4), 826.0, tt.switch(tt.eq(month, 5), 783.3798, tt.switch(tt.eq(month, 6), 1917.8004, tt.switch(tt.eq(month, 7), 2120.6071, tt.switch(tt.eq(month, 8), 1096.7778, tt.switch(tt.eq(month, 9), 2649.4514, tt.switch(tt.eq(month, 10), 4045.6512, 1229.1429))))))))))), tt.switch(tt.eq(month, 0), 1862.5357, tt.switch(tt.eq(month, 1), 1625.2278, tt.switch(tt.eq(month, 2), 1683.6667, tt.switch(tt.eq(month, 3), 1899.2368, tt.switch(tt.eq(month, 4), 2216.875, tt.switch(tt.eq(month, 5), 853.8361, tt.switch(tt.eq(month, 6), 1599.1273, tt.switch(tt.eq(month, 7), 1954.7619, tt.switch(tt.eq(month, 8), 1207.957, tt.switch(tt.eq(month, 9), 2187.6667, tt.switch(tt.eq(month, 10), 1219.7027, 2330.9412)))))))))))), sigma=tt.switch(tt.eq(y, 0), tt.switch(tt.eq(month, 0), 2842.5471, tt.switch(tt.eq(month, 1), 3202.5655, tt.switch(tt.eq(month, 2), 5501.1073, tt.switch(tt.eq(month, 3), 2478.4931, tt.switch(tt.eq(month, 4), 1813.7531, tt.switch(tt.eq(month, 5), 2018.1316, tt.switch(tt.eq(month, 6), 3739.245, tt.switch(tt.eq(month, 7), 2565.8502, tt.switch(tt.eq(month, 8), 2375.6243, tt.switch(tt.eq(month, 9), 3644.7028, tt.switch(tt.eq(month, 10), 11222.2315, 1622.7708))))))))))), tt.switch(tt.eq(month, 0), 3811.0455, tt.switch(tt.eq(month, 1), 2786.9022, tt.switch(tt.eq(month, 2), 1382.0483, tt.switch(tt.eq(month, 3), 2055.7174, tt.switch(tt.eq(month, 4), 2284.5375, tt.switch(tt.eq(month, 5), 1543.8069, tt.switch(tt.eq(month, 6), 2037.8572, tt.switch(tt.eq(month, 7), 2670.6014, tt.switch(tt.eq(month, 8), 1890.1508, tt.switch(tt.eq(month, 9), 1970.479, tt.switch(tt.eq(month, 10), 1542.2663, 4249.0182)))))))))))))
        day = pm.Normal('day', mu=tt.switch(tt.eq(month, 0), campaign*-0.0894+17.6413, tt.switch(tt.eq(month, 1), campaign*0.8894+12.9831, tt.switch(tt.eq(month, 2), campaign*1.1414+13.8885, tt.switch(tt.eq(month, 3), campaign*0.1137+6.2935, tt.switch(tt.eq(month, 4), campaign*0.5229+26.8035, tt.switch(tt.eq(month, 5), campaign*0.58+16.1692, tt.switch(tt.eq(month, 6), campaign*0.1095+10.6508, tt.switch(tt.eq(month, 7), campaign*0.1055+13.6976, tt.switch(tt.eq(month, 8), campaign*0.3315+14.7494, tt.switch(tt.eq(month, 9), campaign*0.3216+17.8643, tt.switch(tt.eq(month, 10), campaign*1.1905+16.6762, campaign*-1.4991+15.0274))))))))))), sigma=tt.switch(tt.eq(month, 0), 6.6706, tt.switch(tt.eq(month, 1), 7.2523, tt.switch(tt.eq(month, 2), 9.1159, tt.switch(tt.eq(month, 3), 5.873, tt.switch(tt.eq(month, 4), 4.0242, tt.switch(tt.eq(month, 5), 8.2104, tt.switch(tt.eq(month, 6), 7.5395, tt.switch(tt.eq(month, 7), 9.0996, tt.switch(tt.eq(month, 8), 7.6423, tt.switch(tt.eq(month, 9), 3.2162, tt.switch(tt.eq(month, 10), 7.6339, 8.691))))))))))))
        duration = pm.Normal('duration', mu=tt.switch(tt.eq(y, 0), campaign*-6.2918+244.3562, campaign*43.7177+453.6438), sigma=tt.switch(tt.eq(y, 0), 209.3662, 379.8245))
        pdays = pm.Normal('pdays', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), day*-1.8824+88.705, day*-0.6779+56.3173), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), day*-2.729+100.9352, day*-0.6742+24.0741), tt.switch(tt.eq(loan, 0), day*-0.0036+0.4453, day*-1.1641+24.7693))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(loan, 0), 115.8252, 99.7195), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(loan, 0), 103.3666, 49.2519), tt.switch(tt.eq(loan, 0), 20.5746, 75.2849))))
        previous = pm.Normal('previous', mu=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), pdays*0.0116+0.1781, pdays*0.0088+0.2565), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), pdays*0.0114+0.3003, pdays*0.0087+0.2251), tt.switch(tt.eq(housing, 0), pdays*0.0155+0.0278, pdays*0.0063+0.0145))), sigma=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(housing, 0), 1.3013, 1.9096), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(housing, 0), 1.6119, 1.4071), tt.switch(tt.eq(housing, 0), 0.2081, 0.2048))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(job, 0), 39.682, tt.switch(tt.eq(job, 1), 40.1564, tt.switch(tt.eq(job, 2), 42.0119, tt.switch(tt.eq(job, 3), 47.3393, tt.switch(tt.eq(job, 4), 40.5408, tt.switch(tt.eq(job, 5), 61.8696, tt.switch(tt.eq(job, 6), 41.4536, tt.switch(tt.eq(job, 7), 38.5707, tt.switch(tt.eq(job, 8), 26.8214, tt.switch(tt.eq(job, 9), 39.4701, tt.switch(tt.eq(job, 10), 40.9062, 48.1053))))))))))), sigma=tt.switch(tt.eq(job, 0), 9.4425, tt.switch(tt.eq(job, 1), 9.0381, tt.switch(tt.eq(job, 2), 8.358, tt.switch(tt.eq(job, 3), 11.0082, tt.switch(tt.eq(job, 4), 9.1924, tt.switch(tt.eq(job, 5), 9.7895, tt.switch(tt.eq(job, 6), 9.407, tt.switch(tt.eq(job, 7), 9.2078, tt.switch(tt.eq(job, 8), 5.2828, tt.switch(tt.eq(job, 9), 8.6716, tt.switch(tt.eq(job, 10), 9.7653, 10.3996))))))))))))
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 263 parameter
#####################
def create_bank_gs_loglikcg(filename="", modelname="bank_gs_loglikcg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        marital = pm.Categorical('marital', p=[0.1168,0.6187,0.2645])
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        housing = pm.Categorical('housing', p=[0.434,0.566])
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=[0.0648,0.14,0.0044,0.0491,0.0327,0.1562,0.1175,0.0108,0.3092,0.086,0.0177,0.0115])
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        campaign = pm.Normal('campaign', mu=2.7936, sigma=3.1098)
        pdays = pm.Normal('pdays', mu=39.7666, sigma=100.1211)
        previous = pm.Normal('previous', mu=pdays*0.0098+0.1541, sigma=1.3827)
        poutcome = pm.Categorical('poutcome', p=[0.1084,0.0436,0.0285,0.8195])
        y = pm.Categorical('y', p=[0.8848,0.1152])
        job = pm.Categorical('job', p=tt.switch(tt.eq(marital, 0), tt.switch(tt.eq(housing, 0), [0.113,0.113,0.0304,0.0348,0.2652,0.1478,0.0217,0.0783,0.0,0.1304,0.0609,0.0043], [0.1443,0.1779,0.0302,0.0168,0.1946,0.0302,0.0336,0.1477,0.0,0.198,0.0268,0.0]), tt.switch(tt.eq(marital, 1), tt.switch(tt.eq(housing, 0), [0.0836,0.1476,0.0486,0.0469,0.2193,0.1169,0.0546,0.0631,0.0051,0.1613,0.0282,0.0247], [0.1034,0.32,0.0462,0.0178,0.1846,0.024,0.0388,0.0997,0.0025,0.1366,0.0258,0.0006]), tt.switch(tt.eq(housing, 0), [0.0929,0.0929,0.0179,0.0179,0.2643,0.0161,0.0464,0.0714,0.1036,0.2232,0.0411,0.0125], [0.1431,0.1918,0.0157,0.0079,0.228,0.0031,0.0236,0.1242,0.0252,0.2248,0.0126,0.0]))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), tt.switch(tt.eq(contact, 0), [0.0253,0.8101,0.1266,0.038], tt.switch(tt.eq(contact, 1), [0.0323,0.871,0.0645,0.0323], [0.0611,0.8397,0.0687,0.0305])), tt.switch(tt.eq(job, 1), tt.switch(tt.eq(contact, 0), [0.3625,0.5927,0.0081,0.0367], tt.switch(tt.eq(contact, 1), [0.5893,0.3214,0.0357,0.0536], [0.396,0.5388,0.015,0.0501])), tt.switch(tt.eq(job, 2), tt.switch(tt.eq(contact, 0), [0.1238,0.3524,0.4667,0.0571], tt.switch(tt.eq(contact, 1), [0.3077,0.3846,0.2308,0.0769], [0.18,0.32,0.42,0.08])), tt.switch(tt.eq(job, 3), tt.switch(tt.eq(contact, 0), [0.4545,0.2576,0.2576,0.0303], tt.switch(tt.eq(contact, 1), [0.4118,0.2941,0.1176,0.1765], [0.6897,0.2069,0.1034,0.0])), tt.switch(tt.eq(job, 4), tt.switch(tt.eq(contact, 0), [0.0267,0.1027,0.8467,0.0239], tt.switch(tt.eq(contact, 1), [0.1176,0.1176,0.7451,0.0196], [0.0676,0.1787,0.7101,0.0435])), tt.switch(tt.eq(job, 5), tt.switch(tt.eq(contact, 0), [0.3517,0.4483,0.1655,0.0345], tt.switch(tt.eq(contact, 1), [0.4048,0.4048,0.0714,0.119], [0.2791,0.5349,0.093,0.093])), tt.switch(tt.eq(job, 6), tt.switch(tt.eq(contact, 0), [0.0603,0.4138,0.5086,0.0172], tt.switch(tt.eq(contact, 1), [0.4,0.4,0.2,0.0], [0.0968,0.4194,0.4516,0.0323])), tt.switch(tt.eq(job, 7), tt.switch(tt.eq(contact, 0), [0.0466,0.9025,0.0297,0.0212], tt.switch(tt.eq(contact, 1), [0.0714,0.8571,0.0714,0.0], [0.0784,0.8235,0.0458,0.0523])), tt.switch(tt.eq(job, 8), tt.switch(tt.eq(contact, 0), [0.0323,0.5161,0.2903,0.1613], tt.switch(tt.eq(contact, 1), [0.0,0.7778,0.0,0.2222], [0.0,0.6154,0.0769,0.3077])), tt.switch(tt.eq(job, 9), tt.switch(tt.eq(contact, 0), [0.0093,0.6716,0.3024,0.0167], tt.switch(tt.eq(contact, 1), [0.0,0.7059,0.2941,0.0], [0.0513,0.6872,0.1949,0.0667])), tt.switch(tt.eq(job, 10), tt.switch(tt.eq(contact, 0), [0.1647,0.5412,0.2706,0.0235], tt.switch(tt.eq(contact, 1), [0.3,0.5,0.2,0.0], [0.2727,0.5152,0.2121,0.0])), tt.switch(tt.eq(contact, 0), [0.2083,0.25,0.25,0.2917], tt.switch(tt.eq(contact, 1), [0.0,0.2,0.2,0.6], [0.2222,0.1111,0.1111,0.5556]))))))))))))))
        day = pm.Normal('day', mu=campaign*0.4064+pdays*-0.0066+15.0422, sigma=8.1157)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 258 parameter
#####################
def create_bank_gs_aiccg(filename="", modelname="bank_gs_aiccg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        job = pm.Categorical('job', p=[0.1057,0.2092,0.0372,0.0248,0.2143,0.0509,0.0405,0.0922,0.0186,0.1699,0.0283,0.0084])
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        housing = pm.Categorical('housing', p=tt.switch(tt.eq(job, 0), [0.3682,0.6318], tt.switch(tt.eq(job, 1), [0.2653,0.7347], tt.switch(tt.eq(job, 2), [0.4405,0.5595], tt.switch(tt.eq(job, 3), [0.6518,0.3482], tt.switch(tt.eq(job, 4), [0.4809,0.5191], tt.switch(tt.eq(job, 5), [0.7826,0.2174], tt.switch(tt.eq(job, 6), [0.5191,0.4809], tt.switch(tt.eq(job, 7), [0.3165,0.6835], tt.switch(tt.eq(job, 8), [0.7619,0.2381], tt.switch(tt.eq(job, 9), [0.4479,0.5521], tt.switch(tt.eq(job, 10), [0.5469,0.4531], [0.9737,0.0263]))))))))))))
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=[0.0648,0.14,0.0044,0.0491,0.0327,0.1562,0.1175,0.0108,0.3092,0.086,0.0177,0.0115])
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        campaign = pm.Normal('campaign', mu=2.7936, sigma=3.1098)
        previous = pm.Normal('previous', mu=0.5426, sigma=1.6936)
        poutcome = pm.Categorical('poutcome', p=[0.1084,0.0436,0.0285,0.8195])
        y = pm.Categorical('y', p=[0.8848,0.1152])
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), tt.switch(tt.eq(contact, 0), [0.0253,0.8101,0.1266,0.038], tt.switch(tt.eq(contact, 1), [0.0323,0.871,0.0645,0.0323], [0.0611,0.8397,0.0687,0.0305])), tt.switch(tt.eq(job, 1), tt.switch(tt.eq(contact, 0), [0.3625,0.5927,0.0081,0.0367], tt.switch(tt.eq(contact, 1), [0.5893,0.3214,0.0357,0.0536], [0.396,0.5388,0.015,0.0501])), tt.switch(tt.eq(job, 2), tt.switch(tt.eq(contact, 0), [0.1238,0.3524,0.4667,0.0571], tt.switch(tt.eq(contact, 1), [0.3077,0.3846,0.2308,0.0769], [0.18,0.32,0.42,0.08])), tt.switch(tt.eq(job, 3), tt.switch(tt.eq(contact, 0), [0.4545,0.2576,0.2576,0.0303], tt.switch(tt.eq(contact, 1), [0.4118,0.2941,0.1176,0.1765], [0.6897,0.2069,0.1034,0.0])), tt.switch(tt.eq(job, 4), tt.switch(tt.eq(contact, 0), [0.0267,0.1027,0.8467,0.0239], tt.switch(tt.eq(contact, 1), [0.1176,0.1176,0.7451,0.0196], [0.0676,0.1787,0.7101,0.0435])), tt.switch(tt.eq(job, 5), tt.switch(tt.eq(contact, 0), [0.3517,0.4483,0.1655,0.0345], tt.switch(tt.eq(contact, 1), [0.4048,0.4048,0.0714,0.119], [0.2791,0.5349,0.093,0.093])), tt.switch(tt.eq(job, 6), tt.switch(tt.eq(contact, 0), [0.0603,0.4138,0.5086,0.0172], tt.switch(tt.eq(contact, 1), [0.4,0.4,0.2,0.0], [0.0968,0.4194,0.4516,0.0323])), tt.switch(tt.eq(job, 7), tt.switch(tt.eq(contact, 0), [0.0466,0.9025,0.0297,0.0212], tt.switch(tt.eq(contact, 1), [0.0714,0.8571,0.0714,0.0], [0.0784,0.8235,0.0458,0.0523])), tt.switch(tt.eq(job, 8), tt.switch(tt.eq(contact, 0), [0.0323,0.5161,0.2903,0.1613], tt.switch(tt.eq(contact, 1), [0.0,0.7778,0.0,0.2222], [0.0,0.6154,0.0769,0.3077])), tt.switch(tt.eq(job, 9), tt.switch(tt.eq(contact, 0), [0.0093,0.6716,0.3024,0.0167], tt.switch(tt.eq(contact, 1), [0.0,0.7059,0.2941,0.0], [0.0513,0.6872,0.1949,0.0667])), tt.switch(tt.eq(job, 10), tt.switch(tt.eq(contact, 0), [0.1647,0.5412,0.2706,0.0235], tt.switch(tt.eq(contact, 1), [0.3,0.5,0.2,0.0], [0.2727,0.5152,0.2121,0.0])), tt.switch(tt.eq(contact, 0), [0.2083,0.25,0.25,0.2917], tt.switch(tt.eq(contact, 1), [0.0,0.2,0.2,0.6], [0.2222,0.1111,0.1111,0.5556]))))))))))))))
        pdays = pm.Normal('pdays', mu=previous*34.1447+21.2405, sigma=81.7426)
        day = pm.Normal('day', mu=campaign*0.4064+pdays*-0.0066+15.0422, sigma=8.1157)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 248 parameter
#####################
def create_bank_gs_biccg(filename="", modelname="bank_gs_biccg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        housing = pm.Categorical('housing', p=[0.434,0.566])
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=[0.0648,0.14,0.0044,0.0491,0.0327,0.1562,0.1175,0.0108,0.3092,0.086,0.0177,0.0115])
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        campaign = pm.Normal('campaign', mu=2.7936, sigma=3.1098)
        previous = pm.Normal('previous', mu=0.5426, sigma=1.6936)
        poutcome = pm.Categorical('poutcome', p=[0.1084,0.0436,0.0285,0.8195])
        y = pm.Categorical('y', p=[0.8848,0.1152])
        job = pm.Categorical('job', p=tt.switch(tt.eq(housing, 0), [0.0897,0.1279,0.0377,0.0372,0.2375,0.0917,0.0484,0.0673,0.0326,0.1753,0.0357,0.0189], [0.118,0.2716,0.0367,0.0152,0.1966,0.0195,0.0344,0.1114,0.0078,0.1657,0.0227,0.0004]))
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), tt.switch(tt.eq(contact, 0), [0.0253,0.8101,0.1266,0.038], tt.switch(tt.eq(contact, 1), [0.0323,0.871,0.0645,0.0323], [0.0611,0.8397,0.0687,0.0305])), tt.switch(tt.eq(job, 1), tt.switch(tt.eq(contact, 0), [0.3625,0.5927,0.0081,0.0367], tt.switch(tt.eq(contact, 1), [0.5893,0.3214,0.0357,0.0536], [0.396,0.5388,0.015,0.0501])), tt.switch(tt.eq(job, 2), tt.switch(tt.eq(contact, 0), [0.1238,0.3524,0.4667,0.0571], tt.switch(tt.eq(contact, 1), [0.3077,0.3846,0.2308,0.0769], [0.18,0.32,0.42,0.08])), tt.switch(tt.eq(job, 3), tt.switch(tt.eq(contact, 0), [0.4545,0.2576,0.2576,0.0303], tt.switch(tt.eq(contact, 1), [0.4118,0.2941,0.1176,0.1765], [0.6897,0.2069,0.1034,0.0])), tt.switch(tt.eq(job, 4), tt.switch(tt.eq(contact, 0), [0.0267,0.1027,0.8467,0.0239], tt.switch(tt.eq(contact, 1), [0.1176,0.1176,0.7451,0.0196], [0.0676,0.1787,0.7101,0.0435])), tt.switch(tt.eq(job, 5), tt.switch(tt.eq(contact, 0), [0.3517,0.4483,0.1655,0.0345], tt.switch(tt.eq(contact, 1), [0.4048,0.4048,0.0714,0.119], [0.2791,0.5349,0.093,0.093])), tt.switch(tt.eq(job, 6), tt.switch(tt.eq(contact, 0), [0.0603,0.4138,0.5086,0.0172], tt.switch(tt.eq(contact, 1), [0.4,0.4,0.2,0.0], [0.0968,0.4194,0.4516,0.0323])), tt.switch(tt.eq(job, 7), tt.switch(tt.eq(contact, 0), [0.0466,0.9025,0.0297,0.0212], tt.switch(tt.eq(contact, 1), [0.0714,0.8571,0.0714,0.0], [0.0784,0.8235,0.0458,0.0523])), tt.switch(tt.eq(job, 8), tt.switch(tt.eq(contact, 0), [0.0323,0.5161,0.2903,0.1613], tt.switch(tt.eq(contact, 1), [0.0,0.7778,0.0,0.2222], [0.0,0.6154,0.0769,0.3077])), tt.switch(tt.eq(job, 9), tt.switch(tt.eq(contact, 0), [0.0093,0.6716,0.3024,0.0167], tt.switch(tt.eq(contact, 1), [0.0,0.7059,0.2941,0.0], [0.0513,0.6872,0.1949,0.0667])), tt.switch(tt.eq(job, 10), tt.switch(tt.eq(contact, 0), [0.1647,0.5412,0.2706,0.0235], tt.switch(tt.eq(contact, 1), [0.3,0.5,0.2,0.0], [0.2727,0.5152,0.2121,0.0])), tt.switch(tt.eq(contact, 0), [0.2083,0.25,0.25,0.2917], tt.switch(tt.eq(contact, 1), [0.0,0.2,0.2,0.6], [0.2222,0.1111,0.1111,0.5556]))))))))))))))
        pdays = pm.Normal('pdays', mu=previous*34.1447+21.2405, sigma=81.7426)
        day = pm.Normal('day', mu=campaign*0.4064+pdays*-0.0066+15.0422, sigma=8.1157)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 249 parameter
#####################
def create_bank_gs_predloglikcg(filename="", modelname="bank_gs_predloglikcg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        marital = pm.Categorical('marital', p=[0.1168,0.6187,0.2645])
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        month = pm.Categorical('month', p=[0.0648,0.14,0.0044,0.0491,0.0327,0.1562,0.1175,0.0108,0.3092,0.086,0.0177,0.0115])
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        campaign = pm.Normal('campaign', mu=2.7936, sigma=3.1098)
        pdays = pm.Normal('pdays', mu=39.7666, sigma=100.1211)
        previous = pm.Normal('previous', mu=pdays*0.0098+0.1541, sigma=1.3827)
        poutcome = pm.Categorical('poutcome', p=[0.1084,0.0436,0.0285,0.8195])
        y = pm.Categorical('y', p=[0.8848,0.1152])
        job = pm.Categorical('job', p=tt.switch(tt.eq(marital, 0), [0.1307,0.1496,0.0303,0.0246,0.2254,0.0814,0.0284,0.1174,0.0,0.1686,0.0417,0.0019], tt.switch(tt.eq(marital, 1), [0.0951,0.2478,0.0472,0.03,0.1991,0.0629,0.0454,0.0844,0.0036,0.1469,0.0268,0.0107], [0.1196,0.1455,0.0167,0.0125,0.245,0.0092,0.0343,0.0995,0.0619,0.2241,0.0259,0.0059])))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), tt.switch(tt.eq(contact, 0), [0.0253,0.8101,0.1266,0.038], tt.switch(tt.eq(contact, 1), [0.0323,0.871,0.0645,0.0323], [0.0611,0.8397,0.0687,0.0305])), tt.switch(tt.eq(job, 1), tt.switch(tt.eq(contact, 0), [0.3625,0.5927,0.0081,0.0367], tt.switch(tt.eq(contact, 1), [0.5893,0.3214,0.0357,0.0536], [0.396,0.5388,0.015,0.0501])), tt.switch(tt.eq(job, 2), tt.switch(tt.eq(contact, 0), [0.1238,0.3524,0.4667,0.0571], tt.switch(tt.eq(contact, 1), [0.3077,0.3846,0.2308,0.0769], [0.18,0.32,0.42,0.08])), tt.switch(tt.eq(job, 3), tt.switch(tt.eq(contact, 0), [0.4545,0.2576,0.2576,0.0303], tt.switch(tt.eq(contact, 1), [0.4118,0.2941,0.1176,0.1765], [0.6897,0.2069,0.1034,0.0])), tt.switch(tt.eq(job, 4), tt.switch(tt.eq(contact, 0), [0.0267,0.1027,0.8467,0.0239], tt.switch(tt.eq(contact, 1), [0.1176,0.1176,0.7451,0.0196], [0.0676,0.1787,0.7101,0.0435])), tt.switch(tt.eq(job, 5), tt.switch(tt.eq(contact, 0), [0.3517,0.4483,0.1655,0.0345], tt.switch(tt.eq(contact, 1), [0.4048,0.4048,0.0714,0.119], [0.2791,0.5349,0.093,0.093])), tt.switch(tt.eq(job, 6), tt.switch(tt.eq(contact, 0), [0.0603,0.4138,0.5086,0.0172], tt.switch(tt.eq(contact, 1), [0.4,0.4,0.2,0.0], [0.0968,0.4194,0.4516,0.0323])), tt.switch(tt.eq(job, 7), tt.switch(tt.eq(contact, 0), [0.0466,0.9025,0.0297,0.0212], tt.switch(tt.eq(contact, 1), [0.0714,0.8571,0.0714,0.0], [0.0784,0.8235,0.0458,0.0523])), tt.switch(tt.eq(job, 8), tt.switch(tt.eq(contact, 0), [0.0323,0.5161,0.2903,0.1613], tt.switch(tt.eq(contact, 1), [0.0,0.7778,0.0,0.2222], [0.0,0.6154,0.0769,0.3077])), tt.switch(tt.eq(job, 9), tt.switch(tt.eq(contact, 0), [0.0093,0.6716,0.3024,0.0167], tt.switch(tt.eq(contact, 1), [0.0,0.7059,0.2941,0.0], [0.0513,0.6872,0.1949,0.0667])), tt.switch(tt.eq(job, 10), tt.switch(tt.eq(contact, 0), [0.1647,0.5412,0.2706,0.0235], tt.switch(tt.eq(contact, 1), [0.3,0.5,0.2,0.0], [0.2727,0.5152,0.2121,0.0])), tt.switch(tt.eq(contact, 0), [0.2083,0.25,0.25,0.2917], tt.switch(tt.eq(contact, 1), [0.0,0.2,0.2,0.6], [0.2222,0.1111,0.1111,0.5556]))))))))))))))
        housing = pm.Categorical('housing', p=tt.switch(tt.eq(job, 0), [0.3682,0.6318], tt.switch(tt.eq(job, 1), [0.2653,0.7347], tt.switch(tt.eq(job, 2), [0.4405,0.5595], tt.switch(tt.eq(job, 3), [0.6518,0.3482], tt.switch(tt.eq(job, 4), [0.4809,0.5191], tt.switch(tt.eq(job, 5), [0.7826,0.2174], tt.switch(tt.eq(job, 6), [0.5191,0.4809], tt.switch(tt.eq(job, 7), [0.3165,0.6835], tt.switch(tt.eq(job, 8), [0.7619,0.2381], tt.switch(tt.eq(job, 9), [0.4479,0.5521], tt.switch(tt.eq(job, 10), [0.5469,0.4531], [0.9737,0.0263]))))))))))))
        day = pm.Normal('day', mu=campaign*0.4064+pdays*-0.0066+15.0422, sigma=8.1157)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 450 parameter
#####################
def create_bank_interiamb_loglikcg(filename="", modelname="bank_interiamb_loglikcg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        campaign = pm.Normal('campaign', mu=2.7936, sigma=3.1098)
        pdays = pm.Normal('pdays', mu=39.7666, sigma=100.1211)
        previous = pm.Normal('previous', mu=pdays*0.0098+0.1541, sigma=1.3827)
        poutcome = pm.Categorical('poutcome', p=[0.1084,0.0436,0.0285,0.8195])
        y = pm.Categorical('y', p=tt.switch(tt.eq(loan, 0), tt.switch(tt.eq(poutcome, 0), [0.8599,0.1401], tt.switch(tt.eq(poutcome, 1), [0.8035,0.1965], tt.switch(tt.eq(poutcome, 2), [0.3659,0.6341], [0.9014,0.0986]))), tt.switch(tt.eq(poutcome, 0), [0.942,0.058], tt.switch(tt.eq(poutcome, 1), [0.8333,0.1667], tt.switch(tt.eq(poutcome, 2), [0.1667,0.8333], [0.9493,0.0507])))))
        age = pm.Normal('age', mu=balance*0.0003+40.751, sigma=10.5402)
        contact = pm.Categorical('contact', p=tt.switch(tt.eq(poutcome, 0), tt.switch(tt.eq(y, 0), [0.911,0.0796,0.0094], [0.9206,0.0794,0.0]), tt.switch(tt.eq(poutcome, 1), tt.switch(tt.eq(y, 0), [0.8679,0.1132,0.0189], [0.8684,0.0789,0.0526]), tt.switch(tt.eq(poutcome, 2), tt.switch(tt.eq(y, 0), [0.8696,0.087,0.0435], [0.8916,0.1084,0.0]), tt.switch(tt.eq(y, 0), [0.568,0.0597,0.3723], [0.7448,0.0801,0.1751])))))
        day = pm.Normal('day', mu=campaign*0.4064+pdays*-0.0066+15.0422, sigma=8.1157)
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(poutcome, 0), [0.1544,0.0447,0.0089,0.1096,0.0649,0.0268,0.0179,0.0089,0.3647,0.1499,0.0336,0.0157], tt.switch(tt.eq(poutcome, 1), [0.1637,0.0643,0.0117,0.076,0.0936,0.0117,0.0526,0.0117,0.345,0.117,0.0175,0.0351], tt.switch(tt.eq(poutcome, 2), [0.114,0.1228,0.0614,0.0702,0.0614,0.0789,0.0439,0.0351,0.1491,0.0789,0.0965,0.0877], [0.0767,0.2597,0.0023,0.0596,0.0356,0.2555,0.0236,0.0129,0.134,0.1151,0.0162,0.0088]))), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(poutcome, 0), [0.0769,0.0513,0.0,0.1538,0.0513,0.0513,0.0513,0.0256,0.359,0.1282,0.0513,0.0], tt.switch(tt.eq(poutcome, 1), [0.0,0.0,0.0476,0.0476,0.1905,0.0,0.0952,0.0476,0.2381,0.2857,0.0476,0.0], tt.switch(tt.eq(poutcome, 2), [0.1538,0.1538,0.0,0.0,0.1538,0.0,0.0,0.2308,0.2308,0.0769,0.0,0.0], [0.0526,0.0702,0.0044,0.0702,0.0439,0.4386,0.0175,0.0219,0.1184,0.114,0.0351,0.0132]))), tt.switch(tt.eq(poutcome, 0), [0.0,0.0,0.0,0.0,0.0,0.0,0.25,0.0,0.0,0.5,0.25,0.0], tt.switch(tt.eq(poutcome, 1), [0.0,0.0,0.0,0.0,0.0,0.2,0.0,0.0,0.2,0.4,0.0,0.2], tt.switch(tt.eq(poutcome, 2), [0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0,0.5], [0.0,0.0046,0.0,0.0,0.0008,0.0206,0.3412,0.0008,0.6238,0.0015,0.003,0.0038]))))))
        housing = pm.Categorical('housing', p=tt.switch(tt.eq(month, 0), [0.2867,0.7133], tt.switch(tt.eq(month, 1), [0.8041,0.1959], tt.switch(tt.eq(month, 2), [0.7,0.3], tt.switch(tt.eq(month, 3), [0.5856,0.4144], tt.switch(tt.eq(month, 4), [0.5946,0.4054], tt.switch(tt.eq(month, 5), [0.5113,0.4887], tt.switch(tt.eq(month, 6), [0.5499,0.4501], tt.switch(tt.eq(month, 7), [0.7755,0.2245], tt.switch(tt.eq(month, 8), [0.128,0.872], tt.switch(tt.eq(month, 9), [0.4242,0.5758], tt.switch(tt.eq(month, 10), [0.8125,0.1875], [0.7115,0.2885]))))))))))))
        job = pm.Categorical('job', p=tt.switch(tt.eq(housing, 0), [0.0897,0.1279,0.0377,0.0372,0.2375,0.0917,0.0484,0.0673,0.0326,0.1753,0.0357,0.0189], [0.118,0.2716,0.0367,0.0152,0.1966,0.0195,0.0344,0.1114,0.0078,0.1657,0.0227,0.0004]))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), [0.0356,0.8222,0.1067,0.0356], tt.switch(tt.eq(job, 1), [0.3901,0.5539,0.0127,0.0433], tt.switch(tt.eq(job, 2), [0.1548,0.3452,0.4345,0.0655], tt.switch(tt.eq(job, 3), [0.5089,0.25,0.1964,0.0446], tt.switch(tt.eq(job, 4), [0.0402,0.1197,0.8122,0.0279], tt.switch(tt.eq(job, 5), [0.3478,0.4565,0.1348,0.0609], tt.switch(tt.eq(job, 6), [0.082,0.4153,0.4809,0.0219], tt.switch(tt.eq(job, 7), [0.06,0.8705,0.0384,0.0312], tt.switch(tt.eq(job, 8), [0.0238,0.5595,0.2262,0.1905], tt.switch(tt.eq(job, 9), [0.0195,0.6771,0.2747,0.0286], tt.switch(tt.eq(job, 10), [0.2031,0.5312,0.25,0.0156], [0.1842,0.2105,0.2105,0.3947]))))))))))))
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), tt.switch(tt.eq(education, 0), [0.1765,0.7059,0.1176], tt.switch(tt.eq(education, 1), [0.1476,0.5547,0.2977], tt.switch(tt.eq(education, 2), [0.0588,0.5294,0.4118], [0.2941,0.5294,0.1765]))), tt.switch(tt.eq(job, 1), tt.switch(tt.eq(education, 0), [0.1003,0.7696,0.1301], tt.switch(tt.eq(education, 1), [0.0706,0.7195,0.2099], tt.switch(tt.eq(education, 2), [0.0,0.25,0.75], [0.122,0.7073,0.1707]))), tt.switch(tt.eq(job, 2), tt.switch(tt.eq(education, 0), [0.0769,0.8462,0.0769], tt.switch(tt.eq(education, 1), [0.0862,0.7759,0.1379], tt.switch(tt.eq(education, 2), [0.1096,0.7534,0.137], [0.0909,0.9091,0.0]))), tt.switch(tt.eq(job, 3), tt.switch(tt.eq(education, 0), [0.1053,0.8421,0.0526], tt.switch(tt.eq(education, 1), [0.1786,0.7143,0.1071], tt.switch(tt.eq(education, 2), [0.0909,0.5455,0.3636], [0.0,0.8,0.2]))), tt.switch(tt.eq(job, 4), tt.switch(tt.eq(education, 0), [0.0513,0.8462,0.1026], tt.switch(tt.eq(education, 1), [0.0948,0.6293,0.2759], tt.switch(tt.eq(education, 2), [0.1309,0.5515,0.3177], [0.1111,0.6296,0.2593]))), tt.switch(tt.eq(job, 5), tt.switch(tt.eq(education, 0), [0.1625,0.8,0.0375], tt.switch(tt.eq(education, 1), [0.1619,0.7905,0.0476], tt.switch(tt.eq(education, 2), [0.2903,0.6452,0.0645], [0.2857,0.6429,0.0714]))), tt.switch(tt.eq(job, 6), tt.switch(tt.eq(education, 0), [0.1333,0.8,0.0667], tt.switch(tt.eq(education, 1), [0.0526,0.7763,0.1711], tt.switch(tt.eq(education, 2), [0.1023,0.6023,0.2955], [0.0,0.75,0.25]))), tt.switch(tt.eq(job, 7), tt.switch(tt.eq(education, 0), [0.2,0.6,0.2], tt.switch(tt.eq(education, 1), [0.146,0.562,0.292], tt.switch(tt.eq(education, 2), [0.0625,0.6875,0.25], [0.2308,0.4615,0.3077]))), tt.switch(tt.eq(job, 8), tt.switch(tt.eq(education, 0), [0.0,0.0,1.0], tt.switch(tt.eq(education, 1), [0.0,0.0851,0.9149], tt.switch(tt.eq(education, 2), [0.0,0.1579,0.8421], [0.0,0.1875,0.8125]))), tt.switch(tt.eq(job, 9), tt.switch(tt.eq(education, 0), [0.1333,0.8,0.0667], tt.switch(tt.eq(education, 1), [0.1288,0.575,0.2962], tt.switch(tt.eq(education, 2), [0.0806,0.4171,0.5024], [0.1364,0.5455,0.3182]))), tt.switch(tt.eq(job, 10), tt.switch(tt.eq(education, 0), [0.2692,0.6538,0.0769], tt.switch(tt.eq(education, 1), [0.1912,0.5588,0.25], tt.switch(tt.eq(education, 2), [0.0625,0.5938,0.3438], [0.0,0.5,0.5]))), tt.switch(tt.eq(education, 0), [0.0,1.0,0.0], tt.switch(tt.eq(education, 1), [0.0,0.875,0.125], tt.switch(tt.eq(education, 2), [0.125,0.25,0.625], [0.0,0.9333,0.0667])))))))))))))))
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 209 parameter
#####################
def create_bank_mmpc_loglikcg(filename="", modelname="bank_mmpc_loglikcg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        job = pm.Categorical('job', p=[0.1057,0.2092,0.0372,0.0248,0.2143,0.0509,0.0405,0.0922,0.0186,0.1699,0.0283,0.0084])
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), [0.0356,0.8222,0.1067,0.0356], tt.switch(tt.eq(job, 1), [0.3901,0.5539,0.0127,0.0433], tt.switch(tt.eq(job, 2), [0.1548,0.3452,0.4345,0.0655], tt.switch(tt.eq(job, 3), [0.5089,0.25,0.1964,0.0446], tt.switch(tt.eq(job, 4), [0.0402,0.1197,0.8122,0.0279], tt.switch(tt.eq(job, 5), [0.3478,0.4565,0.1348,0.0609], tt.switch(tt.eq(job, 6), [0.082,0.4153,0.4809,0.0219], tt.switch(tt.eq(job, 7), [0.06,0.8705,0.0384,0.0312], tt.switch(tt.eq(job, 8), [0.0238,0.5595,0.2262,0.1905], tt.switch(tt.eq(job, 9), [0.0195,0.6771,0.2747,0.0286], tt.switch(tt.eq(job, 10), [0.2031,0.5312,0.25,0.0156], [0.1842,0.2105,0.2105,0.3947]))))))))))))
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        housing = pm.Categorical('housing', p=[0.434,0.566])
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        day = pm.Normal('day', mu=15.9153, sigma=8.2477)
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        campaign = pm.Normal('campaign', mu=day*0.06+duration*-0.0008+2.0424, sigma=3.0635)
        previous = pm.Normal('previous', mu=0.5426, sigma=1.6936)
        poutcome = pm.Categorical('poutcome', p=tt.switch(tt.eq(contact, 0), [0.1544,0.059,0.0394,0.7472], tt.switch(tt.eq(contact, 1), [0.1296,0.0698,0.0432,0.7575], [0.003,0.0038,0.0015,0.9917])))
        y = pm.Categorical('y', p=[0.8848,0.1152])
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(y, 0), [0.0899,0.2153,0.004,0.0657,0.0464,0.2125,0.0169,0.0089,0.1903,0.1246,0.0141,0.0113], [0.1274,0.1755,0.0192,0.0865,0.0337,0.1178,0.0745,0.0385,0.137,0.0865,0.0697,0.0337]), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(y, 0), [0.0545,0.0584,0.0039,0.0817,0.0623,0.358,0.0195,0.0195,0.1751,0.144,0.0195,0.0039], [0.0682,0.1136,0.0227,0.0455,0.0455,0.2273,0.0682,0.1136,0.0909,0.0227,0.1364,0.0455]), tt.switch(tt.eq(y, 0), [0.0,0.004,0.0,0.0,0.0008,0.0206,0.3397,0.0008,0.6239,0.0032,0.0024,0.0048], [0.0,0.0164,0.0,0.0,0.0,0.0328,0.3443,0.0,0.5246,0.0328,0.0328,0.0164]))))
        pdays = pm.Normal('pdays', mu=day*-0.7335+previous*33.9335+33.0283, sigma=81.5282)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 202 parameter
#####################
def create_bank_mmpc_aiccg(filename="", modelname="bank_mmpc_aiccg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        education = pm.Categorical('education', p=[0.15,0.5101,0.2986,0.0414])
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        housing = pm.Categorical('housing', p=[0.434,0.566])
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        day = pm.Normal('day', mu=15.9153, sigma=8.2477)
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        campaign = pm.Normal('campaign', mu=day*0.06+duration*-0.0008+2.0424, sigma=3.0635)
        previous = pm.Normal('previous', mu=0.5426, sigma=1.6936)
        poutcome = pm.Categorical('poutcome', p=[0.1084,0.0436,0.0285,0.8195])
        y = pm.Categorical('y', p=[0.8848,0.1152])
        job = pm.Categorical('job', p=tt.switch(tt.eq(education, 0), [0.0251,0.5442,0.0383,0.0841,0.0575,0.118,0.0221,0.0369,0.0029,0.0221,0.0383,0.0103], tt.switch(tt.eq(education, 1), [0.1704,0.2272,0.0252,0.0121,0.0503,0.0455,0.033,0.1574,0.0204,0.2255,0.0295,0.0035], tt.switch(tt.eq(education, 2), [0.0378,0.0089,0.0541,0.0163,0.583,0.023,0.0652,0.0119,0.0141,0.1563,0.0237,0.0059], [0.0909,0.2193,0.0588,0.0267,0.1444,0.0749,0.0214,0.0695,0.0856,0.1176,0.0107,0.0802]))))
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        contact = pm.Categorical('contact', p=tt.switch(tt.eq(poutcome, 0), [0.9122,0.0796,0.0082], tt.switch(tt.eq(poutcome, 1), [0.868,0.1066,0.0254], tt.switch(tt.eq(poutcome, 2), [0.8837,0.1008,0.0155], [0.5841,0.0615,0.3544]))))
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(y, 0), [0.0899,0.2153,0.004,0.0657,0.0464,0.2125,0.0169,0.0089,0.1903,0.1246,0.0141,0.0113], [0.1274,0.1755,0.0192,0.0865,0.0337,0.1178,0.0745,0.0385,0.137,0.0865,0.0697,0.0337]), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(y, 0), [0.0545,0.0584,0.0039,0.0817,0.0623,0.358,0.0195,0.0195,0.1751,0.144,0.0195,0.0039], [0.0682,0.1136,0.0227,0.0455,0.0455,0.2273,0.0682,0.1136,0.0909,0.0227,0.1364,0.0455]), tt.switch(tt.eq(y, 0), [0.0,0.004,0.0,0.0,0.0008,0.0206,0.3397,0.0008,0.6239,0.0032,0.0024,0.0048], [0.0,0.0164,0.0,0.0,0.0,0.0328,0.3443,0.0,0.5246,0.0328,0.0328,0.0164]))))
        pdays = pm.Normal('pdays', mu=day*-0.7335+previous*33.9335+33.0283, sigma=81.5282)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 209 parameter
#####################
def create_bank_mmpc_biccg(filename="", modelname="bank_mmpc_biccg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        job = pm.Categorical('job', p=[0.1057,0.2092,0.0372,0.0248,0.2143,0.0509,0.0405,0.0922,0.0186,0.1699,0.0283,0.0084])
        marital = pm.Categorical('marital', p=tt.switch(tt.eq(job, 0), [0.1444,0.5565,0.2992], tt.switch(tt.eq(job, 1), [0.0835,0.7326,0.1839], tt.switch(tt.eq(job, 2), [0.0952,0.7857,0.119], tt.switch(tt.eq(job, 3), [0.1161,0.75,0.1339], tt.switch(tt.eq(job, 4), [0.1228,0.5748,0.3024], tt.switch(tt.eq(job, 5), [0.187,0.7652,0.0478], tt.switch(tt.eq(job, 6), [0.082,0.694,0.224], tt.switch(tt.eq(job, 7), [0.1487,0.5659,0.2854], tt.switch(tt.eq(job, 8), [0.0,0.119,0.881], tt.switch(tt.eq(job, 9), [0.1159,0.5352,0.349], tt.switch(tt.eq(job, 10), [0.1719,0.5859,0.2422], [0.0263,0.7895,0.1842]))))))))))))
        education = pm.Categorical('education', p=tt.switch(tt.eq(job, 0), [0.0356,0.8222,0.1067,0.0356], tt.switch(tt.eq(job, 1), [0.3901,0.5539,0.0127,0.0433], tt.switch(tt.eq(job, 2), [0.1548,0.3452,0.4345,0.0655], tt.switch(tt.eq(job, 3), [0.5089,0.25,0.1964,0.0446], tt.switch(tt.eq(job, 4), [0.0402,0.1197,0.8122,0.0279], tt.switch(tt.eq(job, 5), [0.3478,0.4565,0.1348,0.0609], tt.switch(tt.eq(job, 6), [0.082,0.4153,0.4809,0.0219], tt.switch(tt.eq(job, 7), [0.06,0.8705,0.0384,0.0312], tt.switch(tt.eq(job, 8), [0.0238,0.5595,0.2262,0.1905], tt.switch(tt.eq(job, 9), [0.0195,0.6771,0.2747,0.0286], tt.switch(tt.eq(job, 10), [0.2031,0.5312,0.25,0.0156], [0.1842,0.2105,0.2105,0.3947]))))))))))))
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        housing = pm.Categorical('housing', p=[0.434,0.566])
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        duration = pm.Normal('duration', mu=263.9613, sigma=259.8566)
        pdays = pm.Normal('pdays', mu=39.7666, sigma=100.1211)
        previous = pm.Normal('previous', mu=pdays*0.0098+0.1541, sigma=1.3827)
        poutcome = pm.Categorical('poutcome', p=tt.switch(tt.eq(contact, 0), [0.1544,0.059,0.0394,0.7472], tt.switch(tt.eq(contact, 1), [0.1296,0.0698,0.0432,0.7575], [0.003,0.0038,0.0015,0.9917])))
        y = pm.Categorical('y', p=[0.8848,0.1152])
        day = pm.Normal('day', mu=pdays*-0.0078+16.2244, sigma=8.2118)
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(y, 0), [0.0899,0.2153,0.004,0.0657,0.0464,0.2125,0.0169,0.0089,0.1903,0.1246,0.0141,0.0113], [0.1274,0.1755,0.0192,0.0865,0.0337,0.1178,0.0745,0.0385,0.137,0.0865,0.0697,0.0337]), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(y, 0), [0.0545,0.0584,0.0039,0.0817,0.0623,0.358,0.0195,0.0195,0.1751,0.144,0.0195,0.0039], [0.0682,0.1136,0.0227,0.0455,0.0455,0.2273,0.0682,0.1136,0.0909,0.0227,0.1364,0.0455]), tt.switch(tt.eq(y, 0), [0.0,0.004,0.0,0.0,0.0008,0.0206,0.3397,0.0008,0.6239,0.0032,0.0024,0.0048], [0.0,0.0164,0.0,0.0,0.0,0.0328,0.3443,0.0,0.5246,0.0328,0.0328,0.0164]))))
        campaign = pm.Normal('campaign', mu=day*0.06+duration*-0.0008+2.0424, sigma=3.0635)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# 264 parameter
#####################
def create_bank_mmpc_predloglikcg(filename="", modelname="bank_mmpc_predloglikcg", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    model = pm.Model()
    with model:
        age = pm.Normal('age', mu=41.1701, sigma=10.5762)
        marital = pm.Categorical('marital', p=[0.1168,0.6187,0.2645])
        education = pm.Categorical('education', p=[0.15,0.5101,0.2986,0.0414])
        default = pm.Categorical('default', p=[0.9832,0.0168])
        balance = pm.Normal('balance', mu=1422.6578, sigma=3009.6381)
        housing = pm.Categorical('housing', p=[0.434,0.566])
        loan = pm.Categorical('loan', p=[0.8472,0.1528])
        contact = pm.Categorical('contact', p=[0.6406,0.0666,0.2929])
        pdays = pm.Normal('pdays', mu=39.7666, sigma=100.1211)
        previous = pm.Normal('previous', mu=pdays*0.0098+0.1541, sigma=1.3827)
        poutcome = pm.Categorical('poutcome', p=tt.switch(tt.eq(contact, 0), [0.1544,0.059,0.0394,0.7472], tt.switch(tt.eq(contact, 1), [0.1296,0.0698,0.0432,0.7575], [0.003,0.0038,0.0015,0.9917])))
        y = pm.Categorical('y', p=[0.8848,0.1152])
        job = pm.Categorical('job', p=tt.switch(tt.eq(marital, 0), tt.switch(tt.eq(education, 0), [0.038,0.4684,0.0253,0.0759,0.0253,0.1646,0.0253,0.0633,0.0,0.0253,0.0886,0.0], tt.switch(tt.eq(education, 1), [0.2148,0.137,0.0185,0.0185,0.0407,0.063,0.0148,0.1963,0.0,0.2481,0.0481,0.0], tt.switch(tt.eq(education, 2), [0.0194,0.0,0.0516,0.0129,0.6645,0.0581,0.0581,0.0065,0.0,0.1097,0.0129,0.0065], [0.2083,0.2083,0.0417,0.0,0.125,0.1667,0.0,0.125,0.0,0.125,0.0,0.0]))), tt.switch(tt.eq(marital, 1), tt.switch(tt.eq(education, 0), [0.0228,0.5399,0.0418,0.0913,0.0627,0.1217,0.0228,0.0285,0.0,0.0228,0.0323,0.0133], tt.switch(tt.eq(education, 1), [0.1528,0.2642,0.0315,0.014,0.0512,0.0582,0.0413,0.143,0.0028,0.2095,0.0266,0.0049], tt.switch(tt.eq(education, 2), [0.0371,0.0041,0.0757,0.0165,0.597,0.0275,0.0729,0.0151,0.0041,0.121,0.0261,0.0028], [0.0769,0.2479,0.0855,0.0342,0.1453,0.0769,0.0256,0.0513,0.0256,0.1026,0.0085,0.1197]))), tt.switch(tt.eq(education, 0), [0.0274,0.6575,0.0274,0.0411,0.0548,0.0411,0.0137,0.0685,0.0274,0.0137,0.0274,0.0], tt.switch(tt.eq(education, 1), [0.1921,0.1806,0.0131,0.0049,0.0525,0.0082,0.0213,0.1741,0.0706,0.2529,0.0279,0.0016], tt.switch(tt.eq(education, 2), [0.0449,0.0192,0.0214,0.0171,0.5342,0.0043,0.0556,0.0085,0.0342,0.2265,0.0235,0.0107], [0.0652,0.1522,0.0,0.0217,0.1522,0.0217,0.0217,0.087,0.2826,0.1522,0.0217,0.0217]))))))
        day = pm.Normal('day', mu=pdays*-0.0078+16.2244, sigma=8.2118)
        month = pm.Categorical('month', p=tt.switch(tt.eq(contact, 0), tt.switch(tt.eq(y, 0), [0.0899,0.2153,0.004,0.0657,0.0464,0.2125,0.0169,0.0089,0.1903,0.1246,0.0141,0.0113], [0.1274,0.1755,0.0192,0.0865,0.0337,0.1178,0.0745,0.0385,0.137,0.0865,0.0697,0.0337]), tt.switch(tt.eq(contact, 1), tt.switch(tt.eq(y, 0), [0.0545,0.0584,0.0039,0.0817,0.0623,0.358,0.0195,0.0195,0.1751,0.144,0.0195,0.0039], [0.0682,0.1136,0.0227,0.0455,0.0455,0.2273,0.0682,0.1136,0.0909,0.0227,0.1364,0.0455]), tt.switch(tt.eq(y, 0), [0.0,0.004,0.0,0.0,0.0008,0.0206,0.3397,0.0008,0.6239,0.0032,0.0024,0.0048], [0.0,0.0164,0.0,0.0,0.0,0.0328,0.3443,0.0,0.5246,0.0328,0.0328,0.0164]))))
        campaign = pm.Normal('campaign', mu=day*0.0606+1.8292, sigma=3.0697)
        duration = pm.Normal('duration', mu=campaign*-5.714+279.9242, sigma=259.277)
    
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

