from mb_modelbase.core.spflow import SPNModel
from mb_data.allbus import allbus
import spn.structure.leaves.parametric.Parametric as spn_parameter_types
import spn.structure.StatisticalTypes as spn_statistical_types
import numpy as np



m = SPNModel("m", spn_type ='mspn')
df = allbus.categorical_as_strings()

types = {
    'sex': spn_statistical_types.MetaType.BINARY,
    'east-west': spn_statistical_types.MetaType.BINARY,
    'lived_abroad': spn_statistical_types.MetaType.BINARY,
    'age': spn_statistical_types.MetaType.DISCRETE,
    'education': spn_statistical_types.MetaType.DISCRETE,
    'income': spn_statistical_types.MetaType.REAL,
    'happiness': spn_statistical_types.MetaType.DISCRETE,
    'health': spn_statistical_types.MetaType.DISCRETE,
    'orientation': spn_statistical_types.MetaType.DISCRETE
}

m.set_var_types(types)

m.fit(df, type='mspn')
s = m.sample()
s = s.values.tolist()[0]
d = m.density(s)

num = len(m.names)
print(m._maximum())

marg = {
    x: m.copy().marginalize(keep=x)
    for x in m.names
}
age = m.copy().marginalize(keep=['age'])
age.density([31])


