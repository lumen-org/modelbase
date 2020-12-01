#!/bin/bash python3

import os

_dirname = os.path.dirname(__file__)


if __name__ == '__main__':

    from mb.modelbase.utils import model_fitting
    from mb.modelbase import MixableCondGaussianModel
    from mb.data import iris, crabs, mpg

    specs = {
        'mcg_iris': {'class': MixableCondGaussianModel, 'data': iris.mixed(),
                     'fitopts': {'fit_algo': 'map'}},
        'mcg_crabs': {'class': MixableCondGaussianModel, 'data': crabs.mixed(),
                     'fitopts': {'fit_algo': 'map'}},
        'mcg_mpg': {'class': MixableCondGaussianModel, 'data': mpg.cg_4cat3cont(),
                      'fitopts': {'fit_algo': 'map'}},
    }

    models = model_fitting.fit_models(specs)
    model_fitting.save_models(models, directory=os.path.join(_dirname, './example_models'))



