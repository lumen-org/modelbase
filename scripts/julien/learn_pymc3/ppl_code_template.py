import os
import pandas as pd
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model


def code_to_fit(file='{file}', modelname='{modelname}', fit=True):
    # income is gaussian, depends on age
    filepath = os.path.join(os.path.dirname(__file__), '{file}')
    df = pd.read_csv(filepath)
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    model = pm.Model()
    data = None
    with model:
        {pymc3_code}
    m = ProbabilisticPymc3Model(modelname, model)
    m.nr_of_posterior_samples = {sample_size}
    if fit:
        m.fit(df, auto_extend=False)
    return df, m