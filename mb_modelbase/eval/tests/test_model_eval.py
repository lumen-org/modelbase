import mb_modelbase.eval as eval
from mb_modelbase import test_iris
from mb_modelbase import testmodels_pymc3

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # get model
    iris_model = test_iris.mcg_map_model()
    # iris_model.marginalize(remove='species')
    # model = iris_model.copy().marginalize(keep='petal_length')
    model = iris_model.copy().marginalize(keep=['petal_length', 'sepal_length'])

    # model = testmodels_pymc3.coal_mining_desaster()[1]
    # model.marginalize(keep='disasters')

    # compute ppc
    reference, samples = eval.posterior_predictive_check(model, eval.TestQuantities['median'], n=50)
    print(str(reference))
    print(str(samples))

    # make histogram
    plt.hist(samples[0], bins=15)
    plt.axvline(x=reference[0], color='r')
    plt.show()
