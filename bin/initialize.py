#!/usr/bin/env python
# Copyright (C) 2020 , Philipp Lucas, philipp.lucas@dlr.de

import os

import mb.modelbase as mbase
from mb.modelbase.utils import model_fitting
from mb.data import iris, titanic

_dirname = os.path.dirname(__file__)


def _learn_initial_models():
    """Trains some simple probabilistic models and stores them in ./fitted_models."""

    print('Training some simple models ...')
    specs = {
        'iris_cond_gauss': {'class': mbase.MixableCondGaussianModel,
                     'data': iris.mixed(),
                     'fitopts': {'fit_algo': 'map'}},
        'iris_kde': {'class': mbase.KDEModel,
                            'data': iris.mixed(),
                     },
        'titanic_cond_gauss': {'class': mbase.MixableCondGaussianModel,
                        'data': titanic.mixed(),
                        'fitopts': {'fit_algo': 'map'}},
        'titanic_kde':  {'class': mbase.KDEModel,
                        'data': titanic.mixed(),
                        },
    }

    models = model_fitting.fit_models(specs)
    model_fitting.save_models(models, os.path.join(_dirname, './fitted_models'))
    print('...done.')


def _create_empty_config():
    """Creates a mostly empty run_conf.cfg file as a basis for customization."""
    config_content = """
    # custom configuration file for webservice.py
    # see run_conf_defaults.cfg for available settings.
    # note: This file overwrites settings in run_conf_defaults.cfg

    [MODELBASE]
    #model_directory = '<your custom path to models>'
    #cache_enable = False
    #cache_path = /tmp/modelbasecache/    
    """
    print('creating empty config file...')
    try:
        config_file = open(os.path.join(_dirname, './run_conf.cfg'), 'x')
        config_file.write(config_content)
        config_file.close()
    except FileExistsError:
        print('... did not create empty config file as config file "run_conf.cfg" exists already.')
    else:
        print('...done.')


if __name__ == '__main__':
    _create_empty_config()
    _learn_initial_models()
