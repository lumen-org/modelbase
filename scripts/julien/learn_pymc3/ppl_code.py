import os
import pandas as pd
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
def create_fun():
   def code_to_fit(file='../data/allbus_cleaned.csv', modelname='allbus_model_4', fit=True, dtm=None, pp_graph=None):
            # income is gaussian, depends on age
            filepath = os.path.join(os.path.dirname(__file__), '../data/allbus_cleaned.csv')
            df_model_repr = pd.read_csv(filepath)
            df_orig = dtm.backward(df_model_repr, inplace=False)
            if fit:
                modelname = modelname + '_fitted'
            # Set up shared variables

            model = pm.Model()
            data = None
            with model:
                age = pm.Normal('age', mu=52.3588, sigma=17.4324)
                sex = pm.Categorical('sex', p=[0.4803,0.5197])
                eastwest = pm.Categorical('eastwest', p=[0.3399,0.6601])
                spectrum = pm.Categorical('spectrum', p=tt.switch(tt.eq(eastwest, 0), [0.2039,0.3123,0.3032,0.1806], [0.2492,0.412,0.2465,0.0924]))
                educ = pm.Categorical('educ', p=tt.switch(tt.eq(eastwest, 0), [0.0116,0.1677,0.4929,0.0542,0.2735], [0.0033,0.2664,0.2924,0.0957,0.3422]))
                income = pm.Normal('income', mu=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 886.25, 800.0), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 1012.3091, 1191.16), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 1197.55, 1498.2426), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 1271.2273, 2473.15), tt.switch(tt.eq(sex, 0), 1526.6837, 2161.5351))))), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 873.0, 1450.0), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 1021.9716, 1783.88), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 1202.2565, 2258.9), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 1445.087, 2561.4667), tt.switch(tt.eq(sex, 0), 1730.6744, 2906.2763)))))), sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 772.5, 247.4874), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 349.6097, 399.4688), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 619.0254, 791.0166), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 543.7047, 1875.1779), tt.switch(tt.eq(sex, 0), 902.3163, 1332.2688))))), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 980.1056, 212.132), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 598.6517, 922.8269), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 658.6089, 1279.2486), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 709.9401, 1648.815), tt.switch(tt.eq(sex, 0), 1111.0352, 1799.4771)))))))
                health = pm.Categorical('health', p=tt.switch(tt.eq(educ, 0), [0.0714,0.2857,0.2857,0.2857,0.0714], tt.switch(tt.eq(educ, 1), [0.0621,0.1563,0.3635,0.3315,0.0866], tt.switch(tt.eq(educ, 2), [0.028,0.1071,0.2944,0.4197,0.1509], tt.switch(tt.eq(educ, 3), [0.0161,0.0753,0.2796,0.4247,0.2043], [0.0069,0.066,0.1898,0.4512,0.2861])))))
                lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(educ, 0), [0.7857,0.2143], tt.switch(tt.eq(educ, 1), [0.8945,0.1055], tt.switch(tt.eq(educ, 2), [0.8856,0.1144], tt.switch(tt.eq(educ, 3), [0.8011,0.1989], [0.6795,0.3205])))))
                happiness = pm.Normal('happiness', mu=tt.switch(tt.eq(health, 0), age*0.0505+income*0.001+0.9617, tt.switch(tt.eq(health, 1), age*0.0221+income*0.0004+4.9717, tt.switch(tt.eq(health, 2), age*0.0118+income*0.0003+6.3845, tt.switch(tt.eq(health, 3), age*0.0081+income*0.0001+7.5281, age*0.0103+income*0.0001+8.0996)))), sigma=tt.switch(tt.eq(health, 0), 2.2484, tt.switch(tt.eq(health, 1), 1.9394, tt.switch(tt.eq(health, 2), 1.7078, tt.switch(tt.eq(health, 3), 1.3322, 1.2951)))))
                
            m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, probabilistic_program_graph=pp_graph)
            m.nr_of_posterior_samples = 1000
            if fit:
                m.fit(df_orig, auto_extend=False)
            return df_orig, m
   return code_to_fit