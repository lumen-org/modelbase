import pandas as pd
from mb_modelbase.models_core.spflow import SPNModel
from spn.io.Graphics import plot_spn

from mb_data import iris as iris_ds
from mb_data import allbus as allbus_ds
from mb_modelbase.models_core import base
import numpy as np
import matplotlib.pyplot as plt


def spn_allbus():
    """Train and return some spn model based on the allbus data set."""
    allbus = allbus_ds.mixed()

    # Replace all categorical columns with numbers representing the factors
    cats = [x for x in allbus.columns.values if type(allbus[x] == str)]
    for cat in cats:
        allbus[cat] = pd.Categorical(allbus[cat])
        allbus[cat] = allbus[cat].cat.codes

    spn = SPNModel('spn', spn_type='spn')

    spn.set_data(allbus, True)
    spn.fit(var_types=allbus_ds.spn_parameters())

    return allbus, spn


if __name__ == '__main__':

    allbus, spn = spn_allbus()

    spn2 = spn.copy().marginalize(keep='age')

    res = spn2.density([20.7])
    print(res)

    res = spn2.density([25])
    print(res)


    exit(1)

    spn.density([0,0,0,0,0,0,0,0,0,0])

    faces = [ len(np.unique(allbus.iloc[:, i])) for i in range(len(allbus.columns.values))]
    var_types = [spn._nameToVarType[x] for x in spn.names]

    ao = spn.copy().marginalize(['age', 'orientation'])
    cond = ao.copy().condition(base.Condition('age', "==", 34)).marginalize(remove=['age'])
    cond.density([1.5])


    age = spn.copy().marginalize(['age'])
    x = list(range(100))
    y = [age.density([v]) for v in x]
    plt.plot(x,y)
    plt.show()

    ori = spn.copy().marginalize(['orientation'])
    x = list(range(10))
    y = [ori.density([v]) for v in x]
    plt.plot(x,y)
    plt.show()

    ao = spn.copy().marginalize(['age', 'orientation'])
    ase = spn.copy().marginalize(['age', 'orientation'])
    edu = spn.copy().marginalize('education')
    spn.sample()
    age.sample()
    ori.sample()
    ao.sample()

    print([age.density([x]) for x in range(30)])

    ao = spn.copy().marginalize(['age', 'orientation'])
    ao.density([1.0, 1.0])



    test = ao._density_mask.copy()
    np.put(test, np.argwhere(test == 2), [1.4,1.5])

    print(test)










    iris = iris_ds.mixed()
    # iris = pd.read_csv('/home/me/apps/lumen/repos/lumen_data/mb_data/iris/iris.csv')
    iris.species = pd.Categorical(iris.species)
    iris.species = iris.species.cat.codes

    #mspn = SPNModel('Mixed Sum Product Network',
    #                spn_type='mspn',
    #                )

    #mspn.set_data(iris, True)
    #mspn.fit(var_types=iris_ds.spn_metaparameters())

    iris_spn = SPNModel('Mixed Sum Product Network',
                    spn_type='spn',
                    )
    iris_spn.set_data(iris, True)
    iris_spn.fit(var_types=iris_ds.spn_paramaters())

