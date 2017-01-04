# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas





More useful data sets possibly here:
 https://github.com/mwaskom/seaborn-data
"""

import pandas as pd

import seaborn.apionly as sns
import data.adult.adult as adult
import data.heart_disease.heart as heart

from cond_gaussians import ConditionallyGaussianModel
from gaussians import MultiVariateGaussianModel
from categoricals import CategoricalModel

known_models = {
    # dict of what to fit how: {'id': (<model-instance>, <data-frame-to-fit-to>)}

    # categorical models
    'categorical_dummy': (CategoricalModel('categorical_dummy'), pd.read_csv('data/categorical_dummy.csv')),
    'heart': (CategoricalModel('heart'), heart.categorical('data/heart_disease/cleaned.cleveland.data')),
    'adult': (CategoricalModel('adult'), adult.categorical('data/adult/adult.full.cleansed')),

    # multivariate gaussian models
    'iris': (MultiVariateGaussianModel('iris'), sns.load_dataset('iris').iloc[:, 0:-1]),
    'car_crashes': (MultiVariateGaussianModel('car_crashes'), sns.load_dataset('car_crashes').iloc[:, 0:-1]),

    # conditionally gaussian models
    'cg_dummy': (ConditionallyGaussianModel('cg_dummy'), ConditionallyGaussianModel.cg_dummy()),
}


def refit_all_models(verbose=False, include=None, exclude=None):
    """Refits all models as listed below and return a list of these."""

    if include is None:
        include = known_models.keys()
    if exclude is None:
        exclude = []

    # fit it!
    models = []
    for (id_, (model_, df)) in known_models.items():
        if id_ in include and id_ not in exclude:
            if verbose:
                print("Fitting model for data set '" + str(id_) + "' ...", end='')
            models.append(model_.fit(df))
            if verbose:
                print("...done.")

    return models


if __name__ == '__main__':
    import argparse
    import modelbase as mb

    description = """
Allows to refit all 'known' models that are 'registered' in this script.

For this purpose there exists a variable known_models, which is a dictionary that contains all models.
See the code to add new 'known models'!

On execution of this script all known models are fitted again from their data and
stored in the directory that is given as a command line argument.

Examples:
  * 'python refit_all.py -l' lists all known models
  * 'python refit_all.py' refits all models and saves them in the subdirectory 'data_models'
  * 'python refit_all.py --exclude adult' refits all but the adult model
  * 'python refit_all.py --include adult' refits only the adult model"""

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

    modelbase = mb.ModelBase("refitter", load_all=False, model_dir=args.directory)
    for model in refit_all_models(verbose=True, include=args.include, exclude=args.exclude):
        modelbase.add(model)

    print("saving all models to " + str(args.directory) + "...")
    modelbase.save_all_models()
    print("...done")
