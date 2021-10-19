# Copyright (c) 2021 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for the penghuins data set
"""

import seaborn as sns

try:
    import spn.structure.leaves.parametric.Parametric as spn_parameter_types
    import spn.structure.StatisticalTypes as spn_statistical_types
except ImportError:
    pass


def mixed():
    penguins = sns.load_dataset('penguins')
    penguins.dropna(axis=0, inplace=True)
    penguins.columns = ['species', 'island', 'bill length', 'bill depth',
       'flipper length', 'mass', 'sex']
    return penguins


spflow_parameter_types = {
        'species': spn_parameter_types.Categorical,
        'island': spn_parameter_types.Categorical,
        'sex': spn_parameter_types.Categorical,
        'bill length': spn_parameter_types.Gaussian,
        'bill depth':spn_parameter_types.Gaussian ,
        'flipper length': spn_parameter_types.Gaussian,
        'mass': spn_parameter_types.Gaussian,
    }

spflow_meta_types = {
        'species': spn_statistical_types.MetaType.DISCRETE,
        'island': spn_statistical_types.MetaType.DISCRETE,
        'sex': spn_statistical_types.MetaType.BINARY,
        'bill length': spn_statistical_types.MetaType.REAL,
        'bill depth': spn_statistical_types.MetaType.REAL,
        'flipper length': spn_statistical_types.MetaType.REAL,
        'mass': spn_statistical_types.MetaType.REAL,
    }
