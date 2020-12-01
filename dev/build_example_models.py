#!/usr/bin/env python
# Copyright (c) 2017-2020 Philipp Lucas (philipp.lucas@uni-jena.de, philipp.lucas@dlr.de)
"""
This is a collection of model definitions for quick fitting of various models.

@author: Philipp Lucas
"""

import logging
# setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s :: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

import pandas as pd

import mb.data as data
from mb.modelbase.utils.model_fitting import fit_models, save_models

from mb.modelbase.models.cond_gaussians import ConditionallyGaussianModel
from mb.modelbase.models.cond_gaussian_wm import CgWmModel
from mb.modelbase.models.gaussians import MultiVariateGaussianModel
from mb.modelbase.models.categoricals import CategoricalModel
from mb.modelbase.models.mixture_gaussians import MoGModelWithK
from mb.modelbase.models.mixable_cond_gaussian import MixableCondGaussianModel
from mb.modelbase.models.kde_model import KDEModel as KDEModel
from mb.modelbase.models.empirical_model import EmpiricalModel

try:
    from mb.mspn import MSPNModel
except ImportError:
    pass
try:
    from mb.spflow import SPFlowModel
except ImportError:
    pass
try:
    from mb.pymc3 import ProbabilisticPymc3Model
except ImportError:
    pass

""" 
dict of what to fit how:
    {'id': <function with no arguments (a lambda) that returns a <configuration dict>}

<configuration dict>:
    'class': Usually <class-object of model> but can be any function that returns a model when called.
    'data': Optional. The data frame of data to use for fitting. If not spefified the 'class' is expected to return a 
        fitted model.
    'classopts': Optional. A dict passed as keyword-arguments to 'class'.    
    'fitopts': Optional. A dict passed as keyword-arguments to the .fit method of the created model instance.

Idea: the encapsulation in a function prevents that on every execution of the script ALL data is loaded
"""

known_models = {

    # categorical models
    'heart': lambda: ({'class': CategoricalModel, 'data': data.heart.categorical()}),
    'adult': lambda: ({'class': CategoricalModel, 'data': data.adult.categorical()}),

    # multivariate gaussian models
    'iris': lambda: ({'class': MultiVariateGaussianModel, 'data': data.iris.continuous()}),
    'car_crashes': lambda: ({'class': MultiVariateGaussianModel, 'data': data.car_crashes.continuous()}),
    'mvg_dummy_2d': lambda: ({'class': MultiVariateGaussianModel.dummy2d_model}),
    'mvg_crabs': lambda: ({'class': MultiVariateGaussianModel, 'data': data.crabs.continuous()}),

    # mixtures of multivariate gaussians
    # 'faithful': lambda: ({'class': MMVG('faithful'), 'data': df.read_csv()}),

    # conditionally gaussian models
    'cg_crabs': lambda: ({'class': ConditionallyGaussianModel, 'data': data.crabs.mixed()}),
    'cg_olive_oils': lambda: ({'class': ConditionallyGaussianModel, 'data': data.olive_oils.mixed()}),
    #'cg_yeast': lambda: ({'class': ConditionallyGaussianModel, 'data': yeast.mixed()}),
    'cg_iris': lambda: ({'class': ConditionallyGaussianModel, 'data': data.iris.mixed()}),
    'starcraft': lambda: ({'class': ConditionallyGaussianModel, 'data': data.starcraft.cg()}),
    'glass': lambda: ({'class': ConditionallyGaussianModel, 'data': data.glass.mixed()}),
    'abalone': lambda: ({'class': ConditionallyGaussianModel, 'data': data.abalone.data()}),
    'flea': lambda: ({'class': ConditionallyGaussianModel, 'data': data.flea.mixed()}),
    'music': lambda: ({'class': ConditionallyGaussianModel, 'data': data.music.mixed()}),
    'mpg': lambda: ({'class': ConditionallyGaussianModel, 'data': data.mpg.cg()}),
    'census': lambda: ({'class': ConditionallyGaussianModel, 'data': data.zensus.mixed()}),
    'cg_banknotes': lambda: ({'class': ConditionallyGaussianModel, 'data': data.banknotes.mixed()}),

    # conditionally gaussian models with weak marginals
    'cgw_iris': lambda: ({'class': CgWmModel, 'data': data.iris.mixed(), 'fitopts': {'empirical_model_name': 'emp_iris'}}),
    'cgw_crabs': lambda: ({'class': CgWmModel, 'data': data.crabs.mixed()}),
    'cgw_diamonds': lambda: ({'class': CgWmModel, 'data': data.diamonds.mixed()}),
    'cgw_mpg': lambda: ({'class': ConditionallyGaussianModel, 'data': data.mpg.cg()}),

    # mixture of gaussians models
    'mo4g_crabs': lambda: ({'class': MoGModelWithK('mo4g_crabs'), 'classopts': {'k': 4}, 'data': data.crabs.continuous()}),
    'mo10g_crabs': lambda: ({'class': MoGModelWithK('mo10g_crabs'), 'classopts': {'k': 10}, 'data': data.crabs.continuous()}),
    'mo3g_iris': lambda: ({'class': MoGModelWithK('mo3g_iris'), 'classopts': {'k': 3}, 'data': data.iris.continuous()}),

    # mixture of conditional gaussian models
    # mixable cg models
    'mcg_banknotes': lambda: ({'class': MixableCondGaussianModel, 'data': data.banknotes.mixed(), 'fitopts': {'fit_algo': 'map'}}),
    'mcg_crabs_nn': lambda: ({'class': MixableCondGaussianModel, 'data': data.crabs.mixed(), 'fitopts': {'normalized': False}}),
    'mcg_starcraft': lambda: ({'class': MixableCondGaussianModel, 'data': data.starcraft.cg()}),
    'mcg_diamonds_full': lambda: ({'class': MixableCondGaussianModel, 'data': data.diamonds.mixed(), 'fitopts': {'fit_algo': 'full'}}),
    'mcg_diamonds_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.diamonds.mixed(), 'fitopts': {'fit_algo': 'map'}}),

    # mixable cg models
    'mcg_mpg_full': lambda: ({'class': MixableCondGaussianModel, 'data': data.mpg.cg_4cat3cont(), 'fitopts': {'fit_algo': 'full'}}),
    'mcg_mpg_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.mpg.cg_4cat3cont(), 'fitopts': {'fit_algo': 'map', 'empirical_model_name': 'emp_mpg'}}),
    'mcg_mpg_clz': lambda: ({'class': MixableCondGaussianModel, 'data': data.mpg.cg_4cat3cont(), 'fitopts': {'fit_algo': 'clz'}}),

    'mcg_iris_full': lambda: ({'class': MixableCondGaussianModel, 'data': data.iris.mixed(), 'fitopts': {'fit_algo': 'full'}}),
    'mcg_iris_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.iris.mixed(), 'fitopts': {'fit_algo': 'map', 'pci_graph': False}}),
    'mcg_iris_clz': lambda: ({'class': MixableCondGaussianModel, 'data': data.iris.mixed(), 'fitopts': {'fit_algo': 'clz'}}),

    'mcg_crabs_full': lambda: ({'class': MixableCondGaussianModel, 'data': data.crabs.mixed(), 'fitopts': {'fit_algo': 'full'}}),
    'mcg_crabs_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.crabs.mixed(), 'fitopts': {'fit_algo': 'map', 'empirical_model_name': 'emp_crabs'}}),
    'mcg_crabs_clz': lambda: ({'class': MixableCondGaussianModel, 'data': data.crabs.mixed(), 'fitopts': {'fit_algo': 'clz'}}),

    # kde modelle
    'kde_iris': lambda: ({'class': KDEModel, 'data': data.iris.mixed()}),

    # spn models
    #'spn_crabs_i2': lambda: ({'class': SPFlowModel, 'data': data.crabs.continuous(), 'fitopts': {'iterations': 2}}),
    #'spn_crabs_i3': lambda: ({'class': SPFlowModel, 'data': data.crabs.continuous(), 'fitopts': {'iterations': 3}}),
    #'spn_crabs_i4': lambda: ({'class': SPFlowModel, 'data': data.crabs.continuous(), 'fitopts': {'iterations': 4}}),

    #'spn_mpg_i1': lambda: ({'class': SPFlowModel, 'data': mpg.continuous(), 'fitopts': {'iterations': 1}}),
    #'spn_mpg_i2': lambda: ({'class': SPFlowModel, 'data': mpg.continuous(), 'fitopts': {'iterations': 2}}),
    #'spn_mpg_i3': lambda: ({'class': SPFlowModel, 'data': mpg.continuous(), 'fitopts': {'iterations': 3}}),
    #'spn_mpg2_i2': lambda: ({'class': SPFlowModel, 'data': mpg.continuous(None, ['displacement', 'cylinder', 'mpg_city', 'mpg_highway', 'year']), 'fitopts': {'iterations': 2}}),
    #'spn_mpg2_i3': lambda: ({'class': SPFlowModel, 'data': mpg.continuous(['displacement', 'cylinder', 'mpg_city', 'mpg_highway', 'year']), 'fitopts': {'iterations': 3}}),
    #'spn_mpg_i4': lambda: ({'class': SPFlowModel, 'data': mpg.continuous(), 'fitopts': {'iterations': 4}}),
    #'spn_iris_i1': lambda: ({'class': SPFlowModel, 'data': data.iris.continuous(), 'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_iris_cont'}}),
    #'spn_iris_i2': lambda: ({'class': SPFlowModel, 'data': data.iris.continuous(), 'fitopts': {'iterations': 2, 'empirical_model_name': 'emp_iris_cont'}}),
    #'spn_iris_i3': lambda: ({'class': SPFlowModel, 'data': data.iris.continuous(), 'fitopts': {'iterations': 3, 'empirical_model_name': 'emp_iris_cont'}}),
    #'spn_iris_i4': lambda: ({'class': SPFlowModel, 'data': data.iris.continuous(), 'fitopts': {'iterations': 4, 'empirical_model_name': 'emp_iris_cont'}}),

    # other spn models
    #'mspn_iris': lambda: ({'class': MSPNModel, 'data': data.iris.discretized()}),
    #'mspn_iris_2d': lambda: ({'class': MSPNModel, 'data': data.iris.discretized()[['sepal_length', 'petal_length']], 'empirical_model_name': 'emp_iris'}),  # 2d: 2x quant
    #'mspn_iris_3d': lambda: ({'class': MSPNModel, 'data': data.iris.discretized()[['species', 'sepal_length', 'petal_length']]}),  # 3d: 2x quant, 1x cat#

    #'mspn_mpg': lambda: ({'class': MSPNModel, 'data': mpg.cg_4cat3cont()}),
    #'mspn_mpg_t': lambda: ({'class': MSPNModel, 'data': mpg.cg_4cat3cont(), 'classopts': {'threshold': 0.2}}),
    #'mspn_mpg_ti': lambda: ({'class': MSPNModel, 'data': mpg.cg_4cat3cont(), 'classopts': {'threshold': 0.2, 'min_instances_slice': 100}}),
    #'mspn_mpg_i': lambda: ({'class': MSPNModel, 'data': mpg.cg_4cat3cont(), 'classopts': {'min_instances_slice': 100}}),
    # threshold=None, min_instances_slice=

    # credit
    # works, but any mcg model is boring since its way too much regularized.....
    'mcg_credit_map': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/credit/credit.csv',index_col=0), 'fitopts': {'fit_algo': 'map'}}),
    'mcg_credit_clz': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/credit/credit.csv',index_col=0), 'fitopts': {'fit_algo': 'clz'}}),
    #'mspn_credit': lambda: ({'class': MSPNModel, 'data': pd.read_csv('mb_data/credit/credit.csv',index_col=0), 'fitopts': {'fit_algo': 'clz'}}),

    # college
    # works, but non-latent cg models fit badly if there is so few (i.e. 1) categorical dimensions
    'mcg_college_map': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/college/college.csv', index_col=0), 'fitopts': {'fit_algo': 'map'}}),
    'mcg_college_clz': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/college/college.csv', index_col=0), 'fitopts': {'fit_algo': 'clz'}}),
    # seems to fit quite nice, however, SPONModel doesn't support any categorical dimensions
    #'spn_college_i2': lambda: ({'class': SPFlowModel, 'data': pd.read_csv('mb_data/college/college.csv', index_col=0).drop(columns=['Private']), 'fitopts': {'iterations': 2}}),

    # Christoph Saffer
    'bank_chris': lambda: ({'class': MixableCondGaussianModel, 'data': data.bank.mixed_4cat()}),
    'adult_chris': lambda: ({'class': MixableCondGaussianModel, 'data': data.adult.cat4_cont5()}),
    'diamonds_chris': lambda: ({'class': MixableCondGaussianModel, 'data': data.diamonds.mixed_cat3()}),
    'mcg_allbus_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.allbus.spec_cont(), 'fitopts': {'fit_algo': 'map', 'empirical_model_name': 'emp_allbus'}}),
    'mcg_allbus_clz': lambda: ({'class': MixableCondGaussianModel, 'data': data.allbus.spec_cont(), 'fitopts': {'fit_algo': 'clz'}}),
    #'mspn_allbus': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont()}),

    # 'mspn_allbus_i01': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.1}}),
    # 'mspn_allbus_i012': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.12}}),
    # 'mspn_allbus_i014': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.14}}),
    # 'mspn_allbus_i016': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.16}}),
    # 'mspn_allbus_i018': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.18}}),
    # 'mspn_allbus_i02': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.2}}),
    # 'mspn_allbus_i03': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.3}}),
    # 'mspn_allbus_i04': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.4}}),
    # 'mspn_allbus_i05': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.5}}),
    # 'mspn_allbus_i06': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.6}}),
    # 'mspn_allbus_i07': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.7}}),

    # 'mspn_allbus_i01_': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.1, 'min_instances_slice': 50}}),
    # 'mspn_allbus_i02_': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.2, 'min_instances_slice': 50}}),
    # 'mspn_allbus_i03_': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.3, 'min_instances_slice': 50}}),
    # 'mspn_allbus_i04_': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.4, 'min_instances_slice': 50}}),
    # 'mspn_allbus_i05_': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.5, 'min_instances_slice': 50}}),
    # 'mspn_allbus_i06_': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.6, 'min_instances_slice': 50}}),
    # 'mspn_allbus_i07_': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'threshold': 0.7, 'min_instances_slice': 50}}),

    #'mspn_allbus_t10': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 10}}),
    #'mspn_allbus_t25': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 25}}),
    #'mspn_allbus_t50': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 50}}),
    #'mspn_allbus_t75': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 75}}),
    #'mspn_allbus_t100': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 100}}),
    #'mspn_allbus_t125': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 125}}),
    #'mspn_allbus_t150': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 150}}),
    #'mspn_allbus_t175': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 175}}),
    #'mspn_allbus_t200': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cont(), 'classopts': {'min_instances_slice': 200}}),

    'mcg_allbus2_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.allbus.spec_cat(), 'fitopts': {'fit_algo': 'map', 'empirical_model_name': 'emp_allbus2'}}),

    # 'mspn_allbus2': lambda: ({'class': MSPNModel, 'data': data.allbus.spec_cat()}),
    #'spn_allbus_i1': lambda: ({'class': SPFlowModel, 'data': data.allbus.continuous(), 'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_allbus'}}),
    #'spn_allbus_i2': lambda: ({'class': SPFlowModel, 'data': data.allbus.continuous(), 'fitopts': {'iterations': 2, 'empirical_model_name': 'emp_allbus'}}),
    #'spn_allbus_i3': lambda: ({'class': SPFlowModel, 'data': data.allbus.continuous(), 'fitopts': {'iterations': 3, 'empirical_model_name': 'emp_allbus'}}),

    # student models
    # Tristan Kreuzinger
    #'cses_map': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/cses/cses_cleaned.csv'), 'fitopts': {'fit_algo': 'map'}}),
    #'cses_clz': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/cses/cses_small.csv'), 'fitopts': {'fit_algo': 'clz'}}),
    #'cses_sm': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/cses/cses_small.csv')}),

    # Fanli Lin
    'mcg_titanic_full': lambda: ({'class': MixableCondGaussianModel, 'data': data.titanic.mixed(), 'fitopts': {'fit_algo': 'full'}}),
    'mcg_titanic_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.titanic.mixed(), 'fitopts': {'fit_algo': 'map'}}),
    'mcg_titanic_clz': lambda: ({'class': MixableCondGaussianModel, 'data': data.titanic.mixed(), 'fitopts': {'fit_algo': 'clz'}}),
    # 'mspn_titanic': lambda: ({'class': MSPNModel, 'data': data.titanic.mixed()}), # , 'classopts': {'threshold': 0.2, 'min_instances_slice': 100}}),
    # 'mspn_titanic_t02_s100': lambda: ({'class': MSPNModel, 'data': data.titanic.mixed(), 'classopts': {'threshold': 0.2, 'min_instances_slice': 100}}),
    # 'mspn_titanic_t02_s50': lambda: ({'class': MSPNModel, 'data': data.titanic.mixed(), 'classopts': {'threshold': 0.2, 'min_instances_slice': 50}}),
    #'spn_titanic_i1': lambda: ({'class': SPFlowModel, 'data': data.titanic.continuous(), 'fitopts': {'iterations': 1}}),
    #'spn_titanic_i2': lambda: ({'class': SPFlowModel, 'data': data.titanic.continuous(), 'fitopts': {'iterations': 2}}),
    #'spn_titanic_i3': lambda: ({'class': SPFlowModel, 'data': data.titanic.continuous(), 'fitopts': {'iterations': 3}}),
    #'spn_titanic_i4': lambda: ({'class': SPFlowModel, 'data': data.titanic.continuous(), 'fitopts': {'iterations': 4}}),
    #'spn_titanic_i5': lambda: ({'class': SPFlowModel, 'data': data.titanic.continuous(), 'fitopts': {'iterations': 5}}),

    # 'spn_titanic_i4': lambda: ({'class': SPFlowModel, 'data': data.titanic.continuous(), 'fitopts': {'iterations': 4}}),

    # Michael Niebisch

    # empirical model
    'emp_iris': lambda: ({'class': EmpiricalModel, 'data': data.iris.mixed(), 'fitopts': {'empirical_model_name': 'emp_iris', 'pci_graph': True}}),
    'emp_iris_cont': lambda: ({'class': EmpiricalModel, 'data': data.iris.continuous(), 'fitopts': {'empirical_model_name': 'emp_iris_cont'}}),
    'emp_allbus': lambda: ({'class': EmpiricalModel, 'data': data.allbus.spec_cont(), 'fitopts': {'empirical_model_name': 'emp_allbus'}}),
    'emp_allbus2': lambda: ({'class': EmpiricalModel, 'data': data.allbus.spec_cat(), 'fitopts': {'empirical_model_name': 'emp_allbus2'}}),
    'emp_cses': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/cses/old_tristan/cses_cleaned.csv')}),
    'emp_mpg': lambda: ({'class': EmpiricalModel, 'data': data.mpg.cg_4cat3cont()}),
    'emp_titanic': lambda: ({'class': EmpiricalModel, 'data': data.titanic.mixed()}), #, 'fitopts': {'empirical_model_name': 'emp_mpg'}}),
    'emp_crabs': lambda: ({'class': EmpiricalModel, 'data': data.crabs.mixed()}),

    # CHI paper
    'emp_coffeechain': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/coffee_chain/coffee_chain.csv')}),
    'mcg_coffeechain_map': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/coffee_chain/coffee_chain.csv'), 'fitopts': {'fit_algo': 'map'}}),
    'emp_caraccidents': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/airbag.csv')}),

    'emp_activity': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/activity.csv')}),
    'emp_football': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/players_dataset.csv')}),
    'mcg_football_map': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/players_dataset.csv'),'fitopts': {'fit_algo': 'map'}}),

    # Dana Schneider data
    'emp_tom': lambda: ({'class': EmpiricalModel, 'data': data.tom.mixed()}),
    'mcg_tom_map': lambda: ({'class': MixableCondGaussianModel, 'data': data.tom.mixed(), 'fitopts': {'fit_algo': 'map'}}),
    'mcg_tom_clz': lambda: ({'class': MixableCondGaussianModel, 'data': data.tom.mixed(), 'fitopts': {'fit_algo': 'clz'}}),

    # zgraggen generated data set
    'emp_zg_football': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/zgraggen_football.csv')}),
    'mcg_zg_football': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/zgraggen_football.csv'), 'fitopts': {'fit_algo': 'map'}}),

    # pauls parameter stuff
    'emp_paul': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/paul/paul_data.csv')}),
    'mcg_paul': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/paul/paul_data.csv'), 'fitopts': {'fit_algo': 'map'}}),
    'emp_paul_reduced': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/paul/paul_data_reduced.csv')}),
    'mcg_paul_reduced': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/paul/paul_data_reduced.csv'), 'fitopts': {'fit_algo': 'map'}}),

    # neural net stuff
    'emp_nnc-n3000': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/nnc-training_validation_n3000.csv')}),
    'mcg_nnc-n3000': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/nnc-training_validation_n3000.csv'), 'fitopts': {'fit_algo': 'map'}}),
    'emp_nnc-n5000': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/nnc-training_validation_n5000.csv')}),
    'mcg_nnc-n5000': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/nnc-training_validation_n5000.csv'), 'fitopts': {'fit_algo': 'map'}}),
    'emp_nnc-n10000': lambda: ({'class': EmpiricalModel, 'data': pd.read_csv('mb_data/nnc-training_validation_n10000.csv')}),
    'mcg_nnc-n10000': lambda: ({'class': MixableCondGaussianModel, 'data': pd.read_csv('mb_data/nnc-training_validation_n10000.csv'), 'fitopts': {'fit_algo': 'map'}}),

    # jonas gütter MA
    #'pymc3_simplest_test_case': lambda: ({'class': ProbabilisticPymc3Model, 'data': pd.read_csv('mb_data/pymc3/simplest_testcase.csv'), 'classopts': {'model_structure': pymc3_methods.get_models('simplest_test_case')}}),
    #'pymc3_getting_started': lambda: ({'class': ProbabilisticPymc3Model, 'data': pd.read_csv('mb_data/pymc3/getting_started.csv'), 'classopts': {'model_structure': pymc3_methods.get_models('getting_started')}}),
    #'pymc3_coal_mining_disasters': lambda: ({'class': ProbabilisticPymc3Model, 'data': pd.read_csv('mb_data/pymc3/coal_mining_disasters.csv'), 'classopts': {'model_structure': pymc3_methods.get_models('coal_mining_disasters')}}),
    #'pymc3_eight_schools': lambda: ({'class': ProbabilisticPymc3Model, 'data': pd.read_csv('mb_data/pymc3/eight_schools.csv'), 'classopts': {'model_structure': pymc3_methods.get_models('eight_schools')}}),

    # spflow spns
    # 'spflow_allbus2': lambda: ({'class': SPFlowModel, 'data': data.allbus.mixed(),
    #                            'classopts': {'spn_type': 'spn'},
    #                            'fitopts': {'var_types': data.allbus.spn_parameters['christian'],
    #                                        'empirical_model_name': 'emp_allbus2'}}),
    # 'spflow_allbus': lambda: ({'class': SPFlowModel, 'data': data.allbus.mixed(),
    #                            'classopts': {'spn_type': 'spn'},
    #                            'fitopts': {'var_types': data.allbus.spn_parameters['philipp'],
    #                                        'empirical_model_name': 'emp_allbus'}}),
    # 'spflow_iris': lambda: ({'class': SPFlowModel, 'data': data.iris.mixed(),
    #                          'classopts': {'spn_type': 'spn'},
    #                          'fitopts': {'var_types': data.iris.spn_parameters(),
    #                                      'empirical_model_name': 'emp_iris'}}),
    # 'spflow_mpg': lambda: ({'class': SPFlowModel, 'data': mpg.cg_4cat3cont(do_not_change=['cylinder']),
    #                         'classopts': {'spn_type': 'spn'},
    #                         'fitopts': {'var_types': mpg.spflow_parameter_types['version_A'],
    #                                     'empirical_model_name': 'emp_mpg'}}),
    # 'spflow_mpg_autofit': lambda: ({'class': SPFlowModel, 'data': mpg.cg_4cat3cont(do_not_change=['cylinder']),
    #                                 'classopts': {'spn_type': 'mspn'},
    #                                 'fitopts': {'var_types': mpg.spflow_metatypes['version_A'],
    #                                             'empirical_model_name': 'emp_mpg'}}),

    'mcg_eurovis2020': lambda: ({'class': MixableCondGaussianModel,
                                 'data': pd.read_csv('mb_data/eurovis2020/2020-02-06-prestudie.csv').loc[:,
                                         ['wasCorrect', 'timing', 'alpha', 'condition', 'distribution', 'square_1_prob']],'fitopts': {'empirical_model_name': 'emp_eurovis2020'}}),


    # # eurovis 2020
    # 'emp_eurovis2020': lambda: ({'class': EmpiricalModel,
    #                              'data': eurovis2020.mixed(),'fitopts': {'empirical_model_name': 'emp_eurovis2020'}}),

    # # yanira garcia (DLR DW)
    # 'emp_gev': ({'class': EmpiricalModel, 'data': gevgpd.gev()}),
    # 'emp_gpd': ({'class': EmpiricalModel, 'data': gevgpd.gpd()}),

}

if __name__ == '__main__':
    import argparse
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

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--list", help="lists all known models", action="store_true")
    parser.add_argument("-d", "--directory",
                        help="directory to store fitted models in. Defaults to 'data_models'",
                        type=str)
    parser.add_argument("-k", "--keep",
                        help="Boolean flag to disable automatic overwrite of existing models."
                             " Normal behaviour is to overwrite models without asking back.",
                        action="store_true")
    parser.add_argument("-i", "--include",
                        help="list of models to be included. All other will be excluded. "
                             "Defaults to all models.", nargs='+')

    parser.add_argument("-e", "--exclude",
                        help="list of models to be excluded from those otherwise included. "
                             "Defaults to an empty list", nargs='+')

    args = parser.parse_args()

    if args.list:
        logger.info("\n".join(known_models.keys()))
        raise SystemExit()  # exits normally

    if args.directory is None:
        logger.info("using default output directory 'data_models' ... ")
        args.directory = 'data_models'

    if args.keep:
        logger.info("keeping existing models...")
        logger.warning("THIS IS NOT IMPLEMENTED YET!")

    # for debugging:
    debugincl = []
    debugincl += ['mcg_iris_map', 'mcg_titanic_map', 'mcg_mpg_map']
    # debugincl += ['mcg_crabs_map', 'mcg_iris_map', 'mcg_mpg_map', 'pymc3_coal_mining_disasters']

    # overrides any include, IFF args.include is empty, i.e. any command line argument will override these manual settings.
    if args.include is None:
        args.include = debugincl

    models = fit_models(known_models, verbose=True, include=args.include, exclude=args.exclude)
    save_models(models, args.directory)
