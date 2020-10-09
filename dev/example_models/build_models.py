#!/bin/bash python3

if __name__ == '__main__':
    import pandas as pd

    from mb import modelbase as mb
    from mb.modelbase import MixableCondGaussianModel

    specs = {
        'mcg_iris': {'class': MixableCondGaussianModel, 'data': pd.read_csv('./iris.csv'),
                     'fitopts': {'fit_algo': 'map'}},
        'mcg_crabs': {'class': MixableCondGaussianModel, 'data': pd.read_csv('./crabs.csv'),
                     'fitopts': {'fit_algo': 'map'}},
    }

    models = mb.fit_models(specs)
    mb.save_models(models, directory='.')

