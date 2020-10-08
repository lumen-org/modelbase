#!/usr/bin/env python
# Copyright (C) 2020 , Philipp Lucas, philipp.lucas@dlr.de

import os
import sys
import pandas as pd

_dirname = os.path.dirname(__file__)


def _learn_initial_models():
    """Trains some simple probabilistic models and stores them in ./fitted_models."""

    sys.path.append(os.path.join(_dirname, '../doc'))
    import doc.data.titanic as titanic
    import mb_modelbase as mb

    print('Training some simple models ...')
    specs = {
        'mcg_iris': {'class': mb.MixableCondGaussianModel,
                     'data': pd.read_csv(os.path.join(_dirname, '../doc/data/iris.csv')),
                     'fitopts': {'fit_algo': 'map'}},
        'mcg_titanic': {'class': mb.MixableCondGaussianModel,
                        'data': titanic.mixed(),
                        'fitopts': {'fit_algo': 'map'}},
    }

    models = mb.fit_models(specs)
    mb.save_models(models, os.path.join(_dirname, './fitted_models'))
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
