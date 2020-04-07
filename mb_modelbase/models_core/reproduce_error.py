# TODO: what is this? delete?
import numpy as np
import pandas as pd
import pymc3 as pm
import mb_modelbase.models_core.tests.create_PyMC3_testmodels as cr

data, m = cr.create_pymc3_getting_started_model_independent_vars(fit=False)
m._set_data(data, drop_silently=False)
print(m._sample(10)['beta_0'])