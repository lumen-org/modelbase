import pandas as pd
from mb_modelbase.models_core.spflow import SPNModel
from spn.io.Graphics import plot_spn

from mb_data import iris as iris_ds
from mb_data import allbus as allbus_ds
from mb_data import titanic as titanic_ds
from mb_data import mpg as mpg_ds


from mb_modelbase.models_core import base
import numpy as np
import matplotlib.pyplot as plt
import dill
import pickle

from scipy.optimize import minimize


def spn_allbus(model_name='spn_allbus'):
    """Train and return some spn model based on the allbus data set."""
    spn = SPNModel(model_name, spn_type='spn')
    allbus = allbus_ds.mixed()
    spn.set_data(allbus, True)
    spn.fit(var_types=allbus_ds.spn_parameters())

    return allbus, spn


def spn_iris(model_name='spn_iris'):
    """Train and return some spn model based on the iris data set."""
    iris = iris_ds.mixed()
    spn = SPNModel(model_name, spn_type='spn')
    spn.set_data(iris, True)
    spn.fit(var_types=iris_ds.spn_paramaters())
    return iris, spn


#def spn_titanic():
#    """Train and return some spn model based on the titanic data set."""
#    titanic = titanic_ds.mixed()
#    spn = SPNModel('spn_titanic', spn_type='spn')
#    spn.set_data(titanic)
#    spn.fit(var_types=titanic_ds.spn_paramaters())
#    return titanic, spn


#def spn_mpg():
#    """Train and return some spn model based on the mpg data set."""
#    mpg = mpg_ds.mixed()
#    mpg = SPNModel('spn_mpg', spn_type='spn')
#    spn.set_data(mpg)
#    spn.fit(var_types=mpg_ds.spn_paramaters())
#    return mpg, spn

if __name__ == '__main__':
    allbus,spn = spn_allbus()

    def spn_objective(x):
        -1 * spn.density(x.tolist())

    res = minimize(spn_objective,[0,0,0,0,0,0,0,0,0,0],method='Nelder-Mead',tol=1e-6)

    with open('/home/leng_ch/apps/lumen/repos/fitted_models/spn_dill.mdl', 'wb') as f:
        dill.dump(spn, f)

    #p1 = spn.density([0,0,0,0,0,0,0,0,0,0])
    #s = spn.sample().values.tolist()[0]
    #t = spn.data.iloc[0, :].values.tolist()
    #p = spn.density(t)
    #for i in range(1000):
    #    s = spn.sample()
    #    p = spn.density(s.values.tolist()[0])

    # Was ist die Summe der Wahrscheinlichkeiten über alle Ausprägungen eines categoricals
    #inc = spn.copy().marginalize('income')
    #expressions =  np.unique(spn.data['income'])
    #query = ['Female', 'West', 'No', 'Center-Right', 49.0, 4.0, 1210.0, 8.0, 3.0, 6.0]
    #p = []
    #for i in range(len(expressions)):
    #    q_i = query.copy()
    #    q_i[7] = expressions[i]
    #    print(q_i)
    #    p.append(inc.density(q_i))
    #print(np.sum(p))

    #with open('/tmp/test.spn', 'wb') as f:
    #    pickle.dump(spn, f)

    #p = spn.density([53, 'Male', 2, 1240, 'East', 7, 5, 'No', 2, 'Center-Right'])
    #bla = 4
    #p = spn.denstiy(s)
    #
    # allbus, spn = spn_allbus()
    #
    # spn2 = spn.copy().marginalize(keep='age')
    #
    # res = spn2.density([20.7])
    # print(res)
    #
    # res = spn2.density([25])
    # print(res)
    #
    #
    # exit(1)
    #
    # spn.density([1.4, 3.5, 2.5, 2.4 ,3.5,2,7.7,5.4,9.1,1.4])
    #
    # faces = [ len(np.unique(allbus.iloc[:, i])) for i in range(len(allbus.columns.values))]
    # var_types = [spn._nameToVarType[x] for x in spn.names]
    #
    # ao = spn.copy().marginalize(['age', 'orientation'])
    # cond = ao.copy().condition(base.Condition('age', "==", 34)).marginalize(remove=['age'])
    # cond.density([1.5])
    #
    #
    # age = spn.copy().marginalize(['age'])
    # inc = spn.copy().marginalize(['income'])
    # x = list(range(100))
    # y = [age.density([v]) for v in x]
    # plt.plot(x,y)
    # plt.show()
    #
    # ori = spn.copy().marginalize(['orientation'])
    # x = list(range(10))
    # y = [ori.density([v]) for v in x]
    # plt.plot(x,y)
    # plt.show()
    #
    # ao = spn.copy().marginalize(['age', 'orientation'])
    # ase = spn.copy().marginalize(['age', 'orientation'])
    # edu = spn.copy().marginalize('education')
    # spn.sample()
    # age.sample()
    # ori.sample()
    # ao.sample()
    #
    # print([age.density([x]) for x in range(30)])
    #
    # ao = spn.copy().marginalize(['age', 'orientation'])
    # ao.density([1.0, 1.0])
    #
    #
    #
    # test = ao._density_mask.copy()
    # np.put(test, np.argwhere(test == 2), [1.4,1.5])
    #
    # print(test)
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # iris = iris_ds.mixed()
    # # iris = pd.read_csv('/home/me/apps/lumen/repos/lumen_data/mb_data/iris/iris.csv')
    # iris.species = pd.Categorical(iris.species)
    # iris.species = iris.species.cat.codes
    #
    # #mspn = SPNModel('Mixed Sum Product Network',
    # #                spn_type='mspn',
    # #                )
    #
    # #mspn.set_data(iris, True)
    # #mspn.fit(var_types=iris_ds.spn_metaparameters())
    #
    # iris_spn = SPNModel('Mixed Sum Product Network',
    #                 spn_type='spn',
    #                 )
    # iris_spn.set_data(iris, True)
    # iris_spn.fit(var_types=iris_ds.spn_paramaters())

