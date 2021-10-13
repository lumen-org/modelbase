# Copyright (c) 2021 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Data preprocessing and cleansing for the penghuins data set
"""

import seaborn as sns


def mixed():
    penguins = sns.load_dataset('penguins')
    penguins.dropna(axis=0, inplace=True)
    penguins.columns = ['species', 'island', 'bill length', 'bill depth',
       'flipper length', 'mass', 'sex']
    return penguins

