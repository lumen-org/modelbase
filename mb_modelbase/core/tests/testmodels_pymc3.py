import numpy as np
import pandas as pd
import pymc3 as pm
from mb_modelbase.core.models import Model
from mb_modelbase.core.pyMC3_model import ProbabilisticPymc3Model

# TODO: warum braucht man das? was hat theano mit pymc3 zu tun?
import theano


def coal_mining_desaster(modelname='pymc3_coal_mining_disaster_model'):
    """TODO: Documentation here. why is there two returned models? what for?"""
    # data
    disasters = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                          3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                          2, 2, 3, 4, 2, 1, 3, 3, 2, 1, 1, 1, 1, 3, 0, 0,
                          1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                          0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                          3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                          0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    years = np.arange(1851, 1962)
    data = pd.DataFrame({'years': years, 'disasters': disasters})
    years = theano.shared(years)

    with pm.Model() as disaster_model:
        switchpoint = pm.DiscreteUniform('switchpoint', lower=years.min(), upper=years.max(), testval=1900)

        # Priors for pre- and post-switch rates number of disasters
        early_rate = pm.Exponential('early_rate', 1.0)
        late_rate = pm.Exponential('late_rate', 1.0)

        # Allocate appropriate Poisson rates to years before and after current
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

        disasters = pm.Poisson('disasters', rate, observed=data['disasters'])
        # years = pm.Normal('years', mu=data['years'], sd=0.1, observed=data['years'])

    model = ProbabilisticPymc3Model(modelname, disaster_model, shared_vars={'years': years})
    model_fitted = model.copy(name=modelname + '_fitted').fit(data)
    return model, model_fitted


if __name__ == '__main__':
    # path here
    testcasemodel_path = '.'

    # coal mining desaster model
    cm,  cm_fitted = coal_mining_desaster()
    cm.save(testcasemodel_path)
    cm_fitted.save(testcasemodel_path)
