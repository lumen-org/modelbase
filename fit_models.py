#!/usr/bin/env python3
# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

More useful data sets possibly here:
 https://github.com/mwaskom/seaborn-data
"""

import pandas as pd

import seaborn.apionly as sns
import data.adult.adult as adult
import data.heart_disease.heart as heart
import data.crabs.crabs as crabs
import data.olive_oils.olive_oils as olive_oils
import data.yeast.yeast as yeast
import data.starcraft.starcraft as starcraft
import data.glass.glass as glass
import data.abalone.abalone as abalone
import data.flea.flea as flea
import data.music.music as music
import data.mpg.mpg as mpg
import data.census.zensus as zensus
import data.bank.bank as bank
import data.banknotes.banknotes as banknotes
import data.diamonds.diamonds as diamonds

from cond_gaussians import ConditionallyGaussianModel
from cond_gaussian_wm import CgWmModel
from gaussians import MultiVariateGaussianModel
from categoricals import CategoricalModel
from mixture_gaussians import MixtureOfGaussiansModel
from mixture_gaussians import MoGModelWithK

known_models = {
    # dict of what to fit how:
    # {'id': <function with no arguments that returns the pair: (<model-instance>, <data-frame-to-fit-to>)> }
    # the encapsulation in a function prevents that on every execution of the script ALL data is loaded

    # categorical models
    'categorical_dummy': lambda: (CategoricalModel('categorical_dummy'), pd.read_csv('data/categorical_dummy.csv')),
    'heart': lambda: (CategoricalModel('heart'), heart.categorical('data/heart_disease/cleaned.cleveland.data')),
    'adult': lambda: (CategoricalModel('adult'), adult.categorical('data/adult/adult.full.cleansed')),

    # multivariate gaussian models
    'iris': lambda: (MultiVariateGaussianModel('iris'), sns.load_dataset('iris').iloc[:, 0:-1]),
    'car_crashes': lambda: (MultiVariateGaussianModel('car_crashes'), sns.load_dataset('car_crashes').iloc[:, 0:-1]),
    'mvg_dummy_2d': lambda: (MultiVariateGaussianModel.dummy2d_model('mvg_dummy_2d'), None),

    # mixtures of multivariate gaussians
    # 'faithful': lambda: (MMVG('faithful'), df.read_csv('data/faithful/faithful.csv')),

    # conditionally gaussian models
    # 'cg_dummy': lambda: (ConditionallyGaussianModel('cg_dummy'), ConditionallyGaussianModel.cg_dummy()),
    'cg_crabs': lambda: (ConditionallyGaussianModel('cg_crabs'), crabs.mixed('data/crabs/australian-crabs.csv')),
    'cg_olive_oils': lambda: (ConditionallyGaussianModel('cg_olive_oils'),
                              olive_oils.mixed('data/olive_oils/olive.csv')),
    'cg_yeast': lambda: (ConditionallyGaussianModel('cg_yeast'), yeast.mixed('data/yeast/yeast.csv')),
    'cg_iris': lambda: (ConditionallyGaussianModel('cg_iris'), sns.load_dataset('iris')),
    'starcraft': lambda: (ConditionallyGaussianModel('starcraft'), starcraft.cg()),
    'glass': lambda: (ConditionallyGaussianModel('glass'), glass.mixed('data/glass/glass.data.csv')),
    'abalone': lambda: (ConditionallyGaussianModel('abalone'), abalone.cg()),
    'flea': lambda: (ConditionallyGaussianModel('flea'), flea.mixed()),
    'music': lambda: (ConditionallyGaussianModel('music'), music.mixed()),
    'mpg': lambda: (ConditionallyGaussianModel('mpg'), mpg.cg()),
    'census': lambda: (ConditionallyGaussianModel('census'), zensus.mixed()),
    'cg_banknotes': lambda: (ConditionallyGaussianModel('cg_banknotes'), banknotes.mixed('data/banknotes/banknotes.csv')),

    # conditionally gaussian models with weak marginals
    'cgw_iris': lambda: (CgWmModel('cgw_iris'), sns.load_dataset('iris')),
    'cgw_crabs': lambda: (CgWmModel('cgw_crabs'), crabs.mixed('data/crabs/australian-crabs.csv')),
    'cgw_diamonds': lambda: (CgWmModel('cgw_diamonds'), diamonds.mixed('data/diamonds/diamonds.csv')),

    # mixture of gaussians models
    'mo4g_crabs': lambda: (MoGModelWithK('mo4g_crabs', 4), crabs.continuous('data/crabs/australian-crabs.csv')),

    'mo10g_crabs': lambda: (MoGModelWithK('mo10g_crabs', 10), crabs.continuous('data/crabs/australian-crabs.csv'))
}


def refit_all_models(verbose=False, include=None, exclude=None):
    """Refits all models as listed below and return a list of these."""

    if include is None:
        include = known_models.keys()
    if exclude is None:
        exclude = []

    # TODO: refactor below to detect if user specified models to include that don't exist

    # fit it!
    models = []
    for (id_, getter) in known_models.items():
        if id_ in include and id_ not in exclude:
            (model_, df) = getter()
            if verbose:
                print("Fitting model for data set '" + str(id_) + "' ...")
            if df is not None:
                # only fit if data available. otherwise we expect that the model has been fitted elsewhere
                model_.fit(df)
            models.append(model_)
            if verbose:
                print("...done.")

    return models


if __name__ == '__main__':
    import argparse
    import modelbase as mb
    import os

    script_name = os.path.basename(__file__)

    description = """
Allows to refit all 'known' models that are 'registered' in this script.

For this purpose there exists a variable known_models, which is a dictionary that contains all models.
See the code to add new 'known models'!

On execution of this script all known models are fitted again from their data and
stored in the directory that is given as a command line argument.

Examples:
  * '{script_name} -l' lists all known models
  * '{script_name}' refits all models and saves them in the subdirectory 'data_models'
  * '{script_name} --exclude adult' refits all but the adult model
  * '{script_name} --include adult' refits only the adult model""".format(script_name=script_name)

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--list", help="lists all known models", action="store_true")
    parser.add_argument("-d", "--directory", help="directory to store fitted models in. Defaults to 'data_models'",
                        type=str)
    parser.add_argument("-i", "--include", help="list of models to be included. All other will be excluded. "
                                                "Defaults to all models.", nargs='+')
    parser.add_argument("-e", "--exclude", help="list of models to be excluded from those otherwise included. "
                                                "Defaults to an empty list", nargs='+')

    args = parser.parse_args()

    if args.list:
        print("\n".join(known_models.keys()))
        raise SystemExit()  # exits normally

    if args.directory is None:
        print("using default output directory 'data_models' ... ")
        args.directory = 'data_models'

    # for debugging:
    # args.include = ['cg_olive_oils']
    # args.include = ['cg_glass']
    # args.include = ['starcraft']
    # args.include = ['cgw_iris']

    modelbase = mb.ModelBase("refitter", load_all=False, model_dir=args.directory)
    models = refit_all_models(verbose=True, include=args.include, exclude=args.exclude)
    for model in models:
        modelbase.add(model)

    print("saving all models to " + str(args.directory) + "...")
    modelbase.save_all_models()
    print("...done")

