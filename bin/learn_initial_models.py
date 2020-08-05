#!/usr/bin/env python
# Copyright (C) 2020 , Philipp Lucas, philipp.lucas@dlr.de

""" Trains some simple probabilistic models and stores them in ./fitted_models.
"""
if __name__ == '__main__':
    import sys
    import pandas as pd
    sys.path.append('../doc')
    import doc.data.titanic as titanic
    import mb_modelbase as mb

    specs = {
        'mcg_iris': {'class': mb.MixableCondGaussianModel,
                     'data': pd.read_csv('../doc/data/iris.csv'),
                     'fitopts': {'fit_algo': 'map'}},
        'mcg_titanic': {'class': mb.MixableCondGaussianModel,
                        'data': titanic.mixed(),
                        'fitopts': {'fit_algo': 'map'}},
    }

    models = mb.fit_models(specs)
    mb.save_models(models, './fitted_models')
