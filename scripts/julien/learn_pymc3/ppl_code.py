import os
import pandas as pd
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
def create_fun():
   def code_to_fit(file='../data/allbus_cleaned_cleaned.csv', modelname='allbus_graph', fit=True, dtm=None, pp_graph=None):
            # income is gaussian, depends on age
            filepath = os.path.join(os.path.dirname(__file__), '../data/allbus_cleaned_cleaned.csv')
            df_model_repr = pd.read_csv(filepath)
            df_orig = dtm.backward(df_model_repr, inplace=False)
            if fit:
                modelname = modelname + '_fitted'
            # Set up shared variables

            model = pm.Model()
            data = None
            with model:
                age = pm.Categorical('age', p=[0.0044,0.0066,0.0149,0.0127,0.0118,0.0123,0.0132,0.0162,0.0158,0.0145,0.0136,0.0123,0.0118,0.0145,0.0092,0.0162,0.0079,0.0154,0.0114,0.0171,0.0171,0.0193,0.0193,0.0101,0.0285,0.0193,0.0193,0.0228,0.0259,0.025,0.0211,0.0162,0.0224,0.0215,0.0088,0.0145,0.0184,0.0162,0.0145,0.0154,0.0167,0.0215,0.025,0.0211,0.0193,0.0123,0.014,0.0136,0.0114,0.011,0.0132,0.0162,0.0127,0.0154,0.0202,0.0114,0.0101,0.0127,0.0092,0.0079,0.0083,0.0075,0.0048,0.0053,0.0044,0.0035,0.0022,0.0114,0.0035,0.0013,0.0022,0.0013,0.0004,0.0127,0.0092])
                happiness = pm.Categorical('happiness', p=[0.0035,0.0026,0.2268,0.1452,0.0035,0.0184,0.0202,0.0658,0.0605,0.132,0.3215])
                health = pm.Categorical('health', p=tt.switch(tt.eq(happiness, 0), [0.5,0.0,0.375,0.125,0.0], tt.switch(tt.eq(happiness, 1), [0.6667,0.0,0.3333,0.0,0.0], tt.switch(tt.eq(happiness, 2), [0.0019,0.058,0.1741,0.47,0.2959], tt.switch(tt.eq(happiness, 3), [0.0091,0.0574,0.2175,0.3535,0.3625], tt.switch(tt.eq(happiness, 4), [0.375,0.25,0.25,0.0,0.125], tt.switch(tt.eq(happiness, 5), [0.0952,0.3333,0.2381,0.2857,0.0476], tt.switch(tt.eq(happiness, 6), [0.1522,0.3478,0.2826,0.1304,0.087], tt.switch(tt.eq(happiness, 7), [0.1067,0.2933,0.3667,0.1733,0.06], tt.switch(tt.eq(happiness, 8), [0.0435,0.1377,0.4565,0.2971,0.0652], tt.switch(tt.eq(happiness, 9), [0.0332,0.1429,0.3389,0.4086,0.0764], [0.0095,0.0682,0.296,0.4952,0.131])))))))))))
                educ = pm.Categorical('educ', p=tt.switch(tt.eq(health, 0), [0.0154,0.5077,0.3538,0.0462,0.0769], tt.switch(tt.eq(health, 1), [0.0169,0.3502,0.3713,0.0591,0.2025], tt.switch(tt.eq(health, 2), [0.0064,0.3068,0.3847,0.0827,0.2194], tt.switch(tt.eq(health, 3), [0.0043,0.1888,0.3702,0.0848,0.3519], [0.0024,0.1103,0.2974,0.0911,0.4988])))))
                eastwest = pm.Categorical('eastwest', p=tt.switch(tt.eq(educ, 0), [0.6429,0.3571], tt.switch(tt.eq(educ, 1), [0.2448,0.7552], tt.switch(tt.eq(educ, 2), [0.4647,0.5353], tt.switch(tt.eq(educ, 3), [0.2258,0.7742], [0.2916,0.7084])))))
                lived_abroad = pm.Categorical('lived_abroad', p=tt.switch(tt.eq(educ, 0), [0.7857,0.2143], tt.switch(tt.eq(educ, 1), [0.8945,0.1055], tt.switch(tt.eq(educ, 2), [0.8856,0.1144], tt.switch(tt.eq(educ, 3), [0.8011,0.1989], [0.6795,0.3205])))))
                spectrum = pm.Categorical('spectrum', p=tt.switch(tt.eq(eastwest, 0), [0.2039,0.3123,0.3032,0.1806], [0.2492,0.412,0.2465,0.0924]))
                sex = pm.Categorical('sex', p=tt.switch(tt.eq(spectrum, 0), [0.4934,0.5066], tt.switch(tt.eq(spectrum, 1), [0.4745,0.5255], tt.switch(tt.eq(spectrum, 2), [0.5495,0.4505], [0.3226,0.6774]))))
                income = pm.Normal('income', mu=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 886.25, 800.0), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 1012.3091, 1191.16), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 1197.55, 1498.2426), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 1271.2273, 2473.15), tt.switch(tt.eq(sex, 0), 1526.6837, 2161.5351))))), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 873.0, 1450.0), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 1021.9716, 1783.88), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 1202.2565, 2258.9), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 1445.087, 2561.4667), tt.switch(tt.eq(sex, 0), 1730.6744, 2906.2763)))))), sigma=tt.switch(tt.eq(eastwest, 0), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 772.5, 247.4874), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 349.6097, 399.4688), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 619.0254, 791.0166), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 543.7047, 1875.1779), tt.switch(tt.eq(sex, 0), 902.3163, 1332.2688))))), tt.switch(tt.eq(educ, 0), tt.switch(tt.eq(sex, 0), 980.1056, 212.132), tt.switch(tt.eq(educ, 1), tt.switch(tt.eq(sex, 0), 598.6517, 922.8269), tt.switch(tt.eq(educ, 2), tt.switch(tt.eq(sex, 0), 658.6089, 1279.2486), tt.switch(tt.eq(educ, 3), tt.switch(tt.eq(sex, 0), 709.9401, 1648.815), tt.switch(tt.eq(sex, 0), 1111.0352, 1799.4771)))))))
                
            m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, probabilistic_program_graph=pp_graph)
            m.nr_of_posterior_samples = 1000
            if fit:
                m.fit(df_orig, auto_extend=False)
            return df_orig, m
   return code_to_fit