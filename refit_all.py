# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

On execution of this script all known models are fitted again from their data and
stored in the directory that is given as a command line argument.

The variable known_models is a dictionary that contains all these. See the code to
add more 'known models' !

"""

import pandas as pd

import seaborn.apionly as sns
import data.adult.adult as adult
import data.heart_disease.heart as heart

from cond_gaussians import ConditionallyGaussianModel
from gaussians import MultiVariateGaussianModel
from categoricals import CategoricalModel

# list of what to fit how: {'id': (<model-instance>, <data-frame-to-fit-to>)}
known_models = {
    'categorical_dummy': (CategoricalModel('categorical_dummy'), pd.read_csv('data/categorical_dummy.csv')),
    'cg_dummy': (ConditionallyGaussianModel('cg_dummy'), ConditionallyGaussianModel.cg_dummy()),
    'iris': (MultiVariateGaussianModel('iris'), sns.load_dataset('iris').iloc[:, 0:-1]),
    'car_crashes': (MultiVariateGaussianModel('car_crashes'), sns.load_dataset('car_crashes')),
    'heart': (CategoricalModel('heart'), heart.categorical('data/heart_disease/cleaned.cleveland.data')),
    # 'adult': (CategoricalModel('adult'), adult.categorical('data/adult/adult.full.cleansed')),
}


def refit_all_models(verbose=False, include=None, exclude=None):
    """Refits all models as listed below and return a list of these. You may adapt the models to refit by the
    parameters include and exclude:

    Args:
        include: include none but only the models with id as listed in the sequence include.
        exclude: exclude any model with id that matches any element of exclude.
    """

    if include is None:
        include = known_models.keys()

    if exclude is None:
        exclude = set([])

    # fit it!
    models = []
    for (id_, (model_, df)) in known_models.items():
        if id_ in include and id_ not in exclude:
            print("Fitting model for " + str(id_) + "...")
            models.append(model_.fit(df))
            print("...done.")

    return models


if __name__ == '__main__':
    import argparse
    import modelbase as mb

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to store fitted models in. Defaults to 'data_models'", type=str)
    args = parser.parse_args()

    if args.directory is None:
        args.directory = 'data_models'

    modelbase = mb.ModelBase("refitter", load_all=False, model_dir=args.directory)
    for model in refit_all_models(True):
        modelbase.add(model)

    print("saving all models to " + str(args.directory) + "...")
    modelbase.save_all_models()
    print("...done")
