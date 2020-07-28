from time import time
import os
from inspect import getmembers, isfunction
import logging

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier as model1
from sklearn.neural_network import MLPClassifier as model2
from sklearn.neighbors import KNeighborsClassifier as model3
from sklearn.svm import SVC as model4
from sklearn.linear_model import LogisticRegression as model5

from pymc3.exceptions import SamplingError

from mb_modelbase import EmpiricalModel
from mb_modelbase.models_core.mspnmodel import MSPNModel
from mb_modelbase.models_core.spnmodel import SPNModel
from mb_modelbase.models_core.spflow import SPNModel as SPFlowSPNModel
from mb_modelbase.utils.Metrics import cll_allbus, get_results_from_file, generate_happiness_plots, cll_iris, \
    generate_income_plots
from mb_modelbase.utils import fit_models, save_models

import scripts.experiments.allbus as allbus
import scripts.experiments.iris as iris
from scripts.julien.learn_pymc3.PPLModelCreator import PPLModel, generate_new_pymc_file_allbus, \
    generate_new_pymc_file_iris


# directory where the generated models for lumen are saved
fitted_models_directory = "./fitted_models/"

sample_size = 45000

# if you do not want to refit models, set the flag to False

fit_bnlearn = True
fit_spn = False
fit_sklearn = False
fit_hand_tuned = True

# defined models
spn_models = {
    # NIPS 2020 Models
    # EMP
    "allbus": {
        'emp_allbus': lambda: ({'class': EmpiricalModel, 'data': allbus.train(numeric_happy=False),
                                'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        'emp_allbus_continues': lambda: ({'class': EmpiricalModel, 'data': allbus.train_continuous(),
                                          'fitopts': {'empirical_model_name': 'emp_allbus_continues'}}),
        # SPFLOW
        'spflow_allbus_spn': lambda: ({'class': SPFlowSPNModel, 'data': allbus.train(numeric_happy=False),
                                       'classopts': {'spn_type': 'spn'},
                                       'fitopts': {'var_types': allbus.spn_parameters,
                                                   'empirical_model_name': 'emp_allbus'}}),
        'spflow_allbus_mspn': lambda: ({'class': SPFlowSPNModel, 'data': allbus.train(numeric_happy=False),
                                        'classopts': {'spn_type': 'mspn'},
                                        'fitopts': {'var_types': allbus.spn_metatypes,
                                                    'empirical_model_name': 'emp_allbus'}}),
        # MSPN
        'mspn_allbus': lambda: ({'class': MSPNModel, 'data': allbus.train(numeric_happy=False),
                                 'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        'mspn_allbus_threshold': lambda: ({'class': MSPNModel, 'data': allbus.train(numeric_happy=False),
                                           'classopts': {'threshold': 0.1},
                                           'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        'mspn_allbus_min_slice': lambda: ({'class': MSPNModel, 'data': allbus.train(numeric_happy=False),
                                           'classopts': {'min_instances_slice': 150},
                                           'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        # GAUSSPN
        'spn_allbus': lambda: ({'class': SPNModel, 'data': allbus.train_continuous(),
                                'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_allbus_continues'}}),
        'spn_allbus_iterate_3': lambda: ({'class': SPNModel, 'data': allbus.train_continuous(),
                                          'fitopts': {'iterations': 3,
                                                      'empirical_model_name': 'emp_allbus_continues'}}),
    },
    "iris": {
        'emp_iris': lambda: ({'class': EmpiricalModel, 'data': iris.iris(),
                              'fitopts': {'empirical_model_name': 'emp_iris'}}),
        'emp_iris_continuous': lambda: ({'class': EmpiricalModel, 'data': iris.iris(continuous=True),
                                         'fitopts': {'empirical_model_name': 'emp_iris_continuous'}}),
        # SPFLOW
        'spflow_iris_spn': lambda: ({'class': SPFlowSPNModel, 'data': iris.iris(),
                                     'classopts': {'spn_type': 'spn'},
                                     'fitopts': {'var_types': iris.spn_parameters,
                                                 'empirical_model_name': 'emp_iris'}}),
        'spflow_iris_mspn': lambda: ({'class': SPFlowSPNModel, 'data': iris.iris(),
                                      'classopts': {'spn_type': 'mspn'},
                                      'fitopts': {'var_types': iris.spn_metatypes,
                                                  'empirical_model_name': 'emp_iris'}}),
        # MSPN
        'mspn_iris': lambda: ({'class': MSPNModel, 'data': iris.iris(),
                               'fitopts': {'empirical_model_name': 'emp_iris'}}),
        'mspn_iris_threshold': lambda: ({'class': MSPNModel, 'data': iris.iris(),
                                         'classopts': {'threshold': 0.1},
                                         'fitopts': {'empirical_model_name': 'emp_iris'}}),
        'mspn_iris_min_slice': lambda: ({'class': MSPNModel, 'data': iris.iris(),
                                         'classopts': {'min_instances_slice': 25},
                                         'fitopts': {'empirical_model_name': 'emp_iris'}}),
        # GAUSSPN
        'spn_iris_thres_001': lambda: ({'class': SPNModel, 'data': iris.iris(continuous=True),
                                        'classopts': {'corrthresh': 0.01, 'batchsize': 1, 'mergebatch': 1},
                                        'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_iris_continuous'}}),
        'spn_iris_thres_01': lambda: ({'class': SPNModel, 'data': iris.iris(continuous=True),
                                       'classopts': {'corrthresh': 0.1, 'batchsize': 5, 'mergebatch': 5},
                                       'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_iris_continuous'}}),
        'spn_iris_thres_04': lambda: ({'class': SPNModel, 'data': iris.iris(continuous=True),
                                       'classopts': {'corrthresh': 0.4, 'batchsize': 15, 'mergebatch': 15},
                                       'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_iris_continuous'}}),
    }
}

if __name__ == "__main__":
    # generate_happiness_plots(continues_data_file, output_path=os.path.dirname(__file__), one_in_all=True)
    start = time()
    print("Starting experiments...")

    if fit_bnlearn:
        print("Calculate bnlearn models")
        # Since there are random initialisations, you have to restart the calculation different times, until all have fit
        # BNLearn just works on numbers
        data_file_all_numeric = allbus._numeric_data
        # model definition file
        pymc_model_file = "./models/pymc_models_allbus.py"
        # we have beforehand the following discrete variables
        discrete_variables = ['sex', 'eastwest', 'lived_abroad']
        # create a model for different algorithms and scores
        could_not_fit = 1
        iteration = 0
        number_of_fitted = 0
        for algo, score in [("hc", "biccg"), ("fast.iamb", "")]:
            while could_not_fit != 0 and iteration < 10:
                # create a new pymc file
                generate_new_pymc_file_allbus(pymc_model_file, sample_size=sample_size)
                could_not_fit = 0
                iteration += 1
                number_of_fitted = 0
                model_name = f"bnlearn_allbus_{algo}".replace("-", "").replace(".", "")
                ppl_model = PPLModel(model_name, data_file_all_numeric, discrete_variables=discrete_variables,
                                     verbose=False, algo=algo, score=score)
                # the function returns 1 if the model could not be translated
                error = ppl_model.generate_pymc(model_name=model_name, save=True, output_file=pymc_model_file)
                if error:
                    could_not_fit += error
                else:
                    number_of_fitted += 1
        print(f"Could fit: {number_of_fitted} of {could_not_fit + number_of_fitted} bnlearn models")
        # import the beforehand created file
        import scripts.experiments.models.pymc_models_allbus as pymc_allbus_models

        # read all model functions
        functions = [o[1] for o in getmembers(pymc_allbus_models) if isfunction(o[1]) and o[0].startswith("create")]
        create_functions = functions
        # add the hand tuned model
        if fit_hand_tuned:
            from scripts.experiments.models.pymc_model_hand_tuned import create_allbus_model_NH0, \
                create_allbus_model_NH1, create_allbus_model_NH2, create_bnlearn_allbus_tabubiccg_adjusted

            # create_functions.append(create_allbus_model_N_onlymargs)
            # create_functions.append(create_allbus_model_NH0)
            # create_functions.append(create_allbus_model_NH1)
            create_functions.append(create_allbus_model_NH2)
            # create_functions.append(create_bnlearn_allbus_tabubiccg_adjusted)
        # fit all the files
        for index, func in enumerate(create_functions):
            print(f"Calculate metrics for bnlearn model {index + 1} of {len(create_functions)}")
            ready = False
            tried = 0
            try:
                data, m_fitted = func(fit=True)
            except SamplingError as se:
                # the hand tuned model raises an error, after number_of_hand_tuned_fits tries skip this model
                iteration = 0
                print("Try fit the hand tuned model.")
                number_of_hand_tuned_fits = 10
                while iteration < number_of_hand_tuned_fits:
                    try:
                        print(f"Try: {iteration + 1} of {number_of_hand_tuned_fits}")
                        data, m_fitted = func(fit=True)
                        break
                    except SamplingError as se2:
                        iteration += 1
                print("Could not fit the hand tuned model")
                continue
            # create empirical model
            name = "emp_" + m_fitted.name
            m_fitted.set_empirical_model_name(name)
            emp_model = EmpiricalModel(name=name)
            emp_model.fit(df=data)
            # save models
            m_fitted.save(fitted_models_directory)
            emp_model.save(fitted_models_directory)
            if data is not None:
                data.to_csv(os.path.join(fitted_models_directory, m_fitted.name + '.csv'), index=False)

    if fit_spn:
        print("Calculate SPN models")
        models = fit_models(spn_models["allbus"], verbose=True, include=[model for model in spn_models["allbus"]])
        save_models(models, fitted_models_directory)

