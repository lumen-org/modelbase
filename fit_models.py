#!/usr/bin/env python3
# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas
"""

import pandas as pd

import data.adult.adult as adult
import data.heart_disease.heart as heart
import data.crabs.crabs as crabs
import data.car_crashes.car_crashes as car_crashes
import data.iris.iris as iris
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
from mixture_cond_gaussian_wm import MoCGModelWithK
from mixable_cond_gaussian import MixableCondGaussianModel

known_models = {
""" 
dict of what to fit how:
    {'id': <function with no arguments (a lambda) that returns a <configuration dict>}

<configuration dict>:
    'class': Usually <class-object of model> but can be any function that returns a model when called.
    'data': Optional. The data frame of data to use for fitting. If not spefified the 'class' is expected to return a fitted model.
    'classopts': Optional. A dict passed as keyword-arguments to 'class'.    
    'fitopts': Optional. A dict passed as keyword-arguments to the .fit method of the created model instance.
    
Idea: the encapsulation in a function prevents that on every execution of the script ALL data is loaded
"""
    # categorical models
    'categorical_dummy': lambda: ({'class': CategoricalModel, 'data': pd.read_csv('data/categorical_dummy.csv')}),
    'heart': lambda: ({'class': CategoricalModel, 'data': heart.categorical('data/heart_disease/cleaned.cleveland.data')}),
    'adult': lambda: ({'class': CategoricalModel, 'data': adult.categorical('data/adult/adult.full.cleansed')}),

    # multivariate gaussian models
    'iris': lambda: ({'class': MultiVariateGaussianModel, 'data': iris.continuous()}),
    'car_crashes': lambda: ({'class': MultiVariateGaussianModel, 'data': car_crashes.continuous()}),
    'mvg_dummy_2d': lambda: ({'class': MultiVariateGaussianModel.dummy2d_model}),
    'mvg_crabs': lambda: ({'class': MultiVariateGaussianModel, 'data': crabs.continuous('data/crabs/australian-crabs.csv')}),

    # mixtures of multivariate gaussians
    # 'faithful': lambda: ({'class': MMVG('faithful'), 'data': df.read_csv('data/faithful/faithful.csv')}),

    # conditionally gaussian models
    # 'cg_dummy': lambda: ({'class': ConditionallyGaussianModel, 'data': ConditionallyGaussianModel.cg_dummy()}),
    'cg_crabs': lambda: ({'class': ConditionallyGaussianModel, 'data': crabs.mixed('data/crabs/australian-crabs.csv')}),
    'cg_olive_oils': lambda: ({'class': ConditionallyGaussianModel, 'data': olive_oils.mixed('data/olive_oils/olive.csv')}),
    #'cg_yeast': lambda: ({'class': ConditionallyGaussianModel, 'data': yeast.mixed('data/yeast/yeast.csv')}),
    'cg_iris': lambda: ({'class': ConditionallyGaussianModel, 'data': iris.mixed()}),
    'starcraft': lambda: ({'class': ConditionallyGaussianModel, 'data': starcraft.cg()}),
    'glass': lambda: ({'class': ConditionallyGaussianModel, 'data': glass.mixed('data/glass/glass.data.csv')}),
    'abalone': lambda: ({'class': ConditionallyGaussianModel, 'data': abalone.cg()}),
    'flea': lambda: ({'class': ConditionallyGaussianModel, 'data': flea.mixed()}),
    'music': lambda: ({'class': ConditionallyGaussianModel, 'data': music.mixed()}),
    'mpg': lambda: ({'class': ConditionallyGaussianModel, 'data': mpg.cg()}),
    'census': lambda: ({'class': ConditionallyGaussianModel, 'data': zensus.mixed()}),
    'cg_banknotes': lambda: ({'class': ConditionallyGaussianModel, 'data': banknotes.mixed('data/banknotes/banknotes.csv')}),

    # conditionally gaussian models with weak marginals
    'cgw_iris': lambda: ({'class': CgWmModel, 'data': iris.mixed()}),
    'cgw_crabs': lambda: ({'class': CgWmModel, 'data': crabs.mixed('data/crabs/australian-crabs.csv')}),
    'cgw_diamonds': lambda: ({'class': CgWmModel, 'data': diamonds.mixed('data/diamonds/diamonds.csv')}),
    'cgw_mpg': lambda: ({'class': ConditionallyGaussianModel, 'data': mpg.cg()}),

    # mixture of gaussians models
    'mo4g_crabs': lambda: ({'class': MoGModelWithK('mo4g_crabs'), 'classopts': {'k': 4}, 'data': crabs.continuous('data/crabs/australian-crabs.csv')}),
    'mo10g_crabs': lambda: ({'class': MoGModelWithK('mo10g_crabs'), 'classopts': {'k': 10}, 'data': crabs.continuous('data/crabs/australian-crabs.csv')}),
    'mo3g_iris': lambda: ({'class': MoGModelWithK('mo3g_iris'), 'classopts': {'k': 3}, 'data': iris.continuous()}),

    # mixture of conditional gaussian models
    # TODO: _fit missing in that model
    #'mo3cg_iris': lambda: ({'class': MoCGModelWithK('mo3g_iris'), 'classopts': {'k': 3} iris.mixed()}),

    # mixable cg models
    'mcg_crabs': lambda: ({'class': MixableCondGaussianModel,
                           'data': crabs.mixed('data/crabs/australian-crabs.csv')}),
    'mcg_crabs_sm': lambda: ({'class': MixableCondGaussianModel,
                                'data': crabs.mixed('data/crabs/australian-crabs.csv', keep=["sex", "CL", "RW"]),
                                'fitopts': {'normalized': True}}),
    'mcg_crabs_clz': lambda: ({'class': MixableCondGaussianModel,
                           'data': crabs.mixed('data/crabs/australian-crabs.csv'),
                           'fitopts': {'fit_algo': 'clz', 'normalized': True}}),
    'mcg_crabs_clz_sm': lambda: ({'class': MixableCondGaussianModel,
                           'data': crabs.mixed('data/crabs/australian-crabs.csv', keep=["sex", "CL", "RW"]),
                           'fitopts': {'fit_algo': 'clz', 'normalized': True}}),



    'mcg_mpg': lambda: ({'class': MixableCondGaussianModel, 'data': mpg.cg()}),
    'mcg_mpg2': lambda: ({'class': MixableCondGaussianModel,
                          'data': mpg.cg_generic(cols=['trans', 'displ', 'cyl', 'cty', 'hwy', ]),
                          'fitopts': {'fit_algo': 'full', 'normalized': False}}),  # larger model than mcg_mpg
    'mcg_iris': lambda: ({'class': MixableCondGaussianModel, 'data': iris.mixed(),
                          'fitopts': {'fit_algo': 'full', 'normalized': True}}),
    'mpg_starcraft': lambda: ({'class': MixableCondGaussianModel, 'data': starcraft.cg()}),
    'mcg_mpg3': lambda: ({'class': MixableCondGaussianModel,
                          'data': mpg.cg_generic(),  # larger model than mcg_mpg
                          'fitopts': {'fit_algo': 'full', 'normalized': True}}),
    'mcg_mpg_sm': lambda: ({'class': MixableCondGaussianModel,
                            'data': mpg.cg_generic(cols=['displ', 'cyl', 'cty', 'drv']),
                            'fitopts': {'fit_algo': 'full', 'normalized': True}}),
    'mcg_diamonds': lambda: ({'class': MixableCondGaussianModel,
                              'data': diamonds.mixed('data/diamonds/diamonds.csv')}),
}


def refit_all_models(verbose=False, include=None, exclude=None):
    """Refits all models as listed below and return a list of these."""

    if include is None:
        include = known_models.keys()
    if exclude is None:
        exclude = set()
    include = set(include)
    fitted = set()  # set of names of models that we've fitted

    # fit it!
    models = []
    for (id_, getter) in known_models.items():
        if id_ in include and id_ not in exclude:
            config = {'classopts': {}, 'fitopts': {}, 'data': None}
            config.update(getter())  # getter() returns the dict that is setup above in known models
            Modelclass = config['class']
            df = config['data']
            if verbose:
                print("Fitting model for data set '" + str(id_) + "' ...")
            model = Modelclass(id_, **config['classopts'])
            if df is not None:
                # only fit if data available. otherwise we expect that the model has been fitted elsewhere
                model.fit(df, **config['fitopts'])
            models.append(model)
            if verbose:
                print("... done.")
            fitted.add(id_)

    # check if all the models in include were found and fitted
    print("Fitted " + str(len(fitted)) + " models in total.")
    if include != fitted:
        print("ERROR: Not all models that you '--include'd were found in the known_models. The missing models are:\n " + str(include - fitted))
        print("I continue with the models I fitted: \n"
              + "<none>" if len(fitted) == 0 else str(fitted))
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
    # args.include = ['mcg_iris']
    # args.include = ['cg_olive_oils']
    # args.include = ['cg_glass']
    # args.include = ['mcg_diamonds']
    # args.include = ['cg_iris']
    # args.include = ['mvg_dummy_2d']

    modelbase = mb.ModelBase("refitter", load_all=False, model_dir=args.directory)
    models = refit_all_models(verbose=True, include=args.include, exclude=args.exclude)
    for model in models:
        modelbase.add(model)

    print("saving all models to " + str(args.directory) + "...")
    modelbase.save_all_models()
    print("...done")

