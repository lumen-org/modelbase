import pandas as pd
from mb_modelbase.models_core.spflow import SPNModel
from spn.structure.leaves.parametric.Parametric import Categorical, Bernoulli, Poisson
from spn.structure.StatisticalTypes import MetaType
from spn.io.Graphics import plot_spn

from mb_data import iris as iris_ds
from mb_data import allbus as allbus_ds

allbus = allbus_ds.mixed()

# Replace all categorical columns with numbers representing the factors
cats = [x for x in allbus.columns.values if type(allbus[x] == str)]
for cat in cats:
    allbus[cat] = pd.Categorical(allbus[cat])
    allbus[cat] = allbus[cat].cat.codes

allbus_var_types = {
    'sex': Bernoulli,
    'east-west': Bernoulli,
    'lived_abroad': Bernoulli,
    'orientation_cat': Categorical,
    'age': Poisson,
    'education': Categorical,
    'income': Poisson,
    'happiness': Categorical,
    'health': Categorical,
    'orientation': Categorical
}

spn = SPNModel('Sum Product Network',
               spn_type='spn',
               var_types=allbus_var_types
               )

spn.set_data(allbus, True)
spn.fit()

age = spn.copy().marginalize(['age'])
ori = spn.copy().marginalize(['orientation'])
ao = spn.copy().marginalize(['age', 'orientation'])

spn.sample()
age.sample()
ori.sample()
ao.sample()

print([age.density([x]) for x in range(30)])

ao = spn.copy().marginalize(['age', 'orientation'])
ao.density([1.0, 1.0])

iris = iris_ds.mixed()
# iris = pd.read_csv('/home/me/apps/lumen/repos/lumen_data/mb_data/iris/iris.csv')
iris.species = pd.Categorical(iris.species)
iris.species = iris.species.cat.codes

iris_meta_types = {
    'sepal_length': MetaType.REAL,
    'sepal_width': MetaType.REAL,
    'petal_length': MetaType.REAL,
    'petal_width': MetaType.REAL,
    'species': MetaType.DISCRETE
}

mspn = SPNModel('Mixed Sum Product Network',
                spn_type='mspn',
                var_types=iris_meta_types
                )

mspn.set_data(iris, True)
mspn.fit()
