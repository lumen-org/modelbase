import mb_modelbase.model_eval as eval
from mb_modelbase import test_iris
from mb_modelbase import testmodels_pymc3

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # get model
    iris_model = test_iris.mcg_map_model()
    # iris_model.marginalize(remove='species')
    model = iris_model.copy().marginalize(keep='petal_length')

    # model = testmodels_pymc3.coal_mining_desaster()[1]
    # model.marginalize(keep='disasters')

    # compute ppc
    reference, samples = eval.posterior_predictive_check(model, np.min, n=100)

    # make histogram
    plt.hist(samples, bins=15)
    plt.axvline(x=reference, color='r')
    plt.show()
