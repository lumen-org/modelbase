from time import time
import os
from inspect import getmembers, isfunction
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
from mb_modelbase.utils.Metrics import cll_allbus, get_results_from_file, generate_happiness_plots, cll_iris
from mb_modelbase.utils import fit_models, save_models

import scripts.experiments.allbus as allbus
import scripts.experiments.iris as iris
from scripts.julien.learn_pymc3.PPLModelCreator import PPLModel, generate_new_pymc_file_allbus, generate_new_pymc_file_iris

# file where results from the experiments are saved
result_file = "allbus_results.dat"

# file for the density prediction values of happiness
continues_data_file = "allbus_happiness_values.dat"

# directory where the generated models for lumen are saved
fitted_models_directory = "./fitted_models/"

# defined models
spn_models = {
    # NIPS 2020 Models
    # EMP
    "allbus": {
        'emp_allbus': lambda: ({'class': EmpiricalModel, 'data': allbus.train(discrete_happy=False),
                                'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        'emp_allbus_continues': lambda: ({'class': EmpiricalModel, 'data': allbus.train_continuous(),
                                          'fitopts': {'empirical_model_name': 'emp_allbus_continues'}}),
        # SPFLOW
        'spflow_allbus_spn': lambda: ({'class': SPFlowSPNModel, 'data': allbus.train(discrete_happy=False),
                                       'classopts': {'spn_type': 'spn'},
                                       'fitopts': {'var_types': allbus.spn_parameters,
                                                   'empirical_model_name': 'emp_allbus'}}),
        'spflow_allbus_mspn': lambda: ({'class': SPFlowSPNModel, 'data': allbus.train(discrete_happy=False),
                                        'classopts': {'spn_type': 'mspn'},
                                        'fitopts': {'var_types': allbus.spn_metatypes,
                                                    'empirical_model_name': 'emp_allbus'}}),
        # MSPN
        'mspn_allbus': lambda: ({'class': MSPNModel, 'data': allbus.train(discrete_happy=False),
                                 # 'classopts': {'threshold': 0.1, 'min_instances_slice': 50},
                                 'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        'mspn_allbus_threshold': lambda: ({'class': MSPNModel, 'data': allbus.train(discrete_happy=False),
                                           'classopts': {'threshold': 0.1},
                                           'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        'mspn_allbus_min_slice': lambda: ({'class': MSPNModel, 'data': allbus.train(discrete_happy=False),
                                           'classopts': {'min_instances_slice': 150},
                                           'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        # GAUSSPN
        'spn_allbus': lambda: ({'class': SPNModel, 'data': allbus.train_continuous(),
                                'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_allbus_continues'}}),
        'spn_allbus_iterate_3': lambda: ({'class': SPNModel, 'data': allbus.train_continuous(),
                                          'fitopts': {'iterations': 3, 'empirical_model_name': 'emp_allbus_continues'}}),
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
                                           'fitopts': {'empirical_model_name': 'emp_allbus'}}),
        'mspn_iris_min_slice': lambda: ({'class': MSPNModel, 'data': iris.iris(),
                                           'classopts': {'min_instances_slice': 150},
                                           'fitopts': {'empirical_model_name': 'emp_iris'}}),
        # GAUSSPN
        'spn_iris': lambda: ({'class': SPNModel, 'data': iris.iris(continuous=True),
                                'fitopts': {'iterations': 1, 'empirical_model_name': 'emp_iris_continuous'}}),
        'spn_iris_iterate_3': lambda: ({'class': SPNModel, 'data': iris.iris(continuous=True),
                                          'fitopts': {'iterations': 3, 'empirical_model_name': 'emp_iris_continuous'}}),
    }
}

fit_spn = True
fit_bnlearn = True
fit_sklearn = True

data_set = "iris"

if __name__ == "__main__":
    start = time()
    print("Starting experiments...")

    if data_set == "allbus":
        print("Resetting results")
        with open(result_file, "w") as f:
            f.write(
                f"modelname, h-mape, h-mae, sex-acc, sex-mae, ewa-acc, ew-mae, la-acc, "
                f"la-mae, q1, q2, q3, q4\n")

        print("Resetting continues data results for happiness")
        # create contines data file
        with open(continues_data_file, "w+") as f:
            f.write(f"model,{','.join([str(i) for i in np.arange(0, 10, 0.1)])}\n")

    if fit_spn:
        print("Calculate SPN models")
        models = fit_models(spn_models[data_set], verbose=True, include=spn_models[data_set])
        print("Calculate SPN model scores (skip emp)")
        number_of_spn_models = len([m for m in models if not m.startswith("emp")])
        for index, (model_name, property) in enumerate(models.items()):
            print(f"Calculate SPN model {index+1} of {number_of_spn_models}")
            if not model_name.startswith("emp"):
                if not model_name.startswith("spn_"):
                    if data_set == "allbus":
                        cll_allbus(property["model"], allbus.test(discrete_happy=False), result_file, continues_data_file)
        save_models(models, fitted_models_directory)

    if fit_bnlearn:
        print("Calculate bnlearn models")
        # Since there are random initialisations, you have to restart the calculation different times, until all have fit
        # BNLearn just works on numbers
        if data_set == "allbus":
            data_file_all_numeric = allbus._numeric_data
            # model definition file
            pymc_model_file = "./models/pymc_models_allbus.py"
            # we have beforehand the following discrete variables
            discrete_variables = ['sex', 'eastwest', 'lived_abroad']
        elif data_set == "iris":
            data_file_all_numeric = iris._numeric_data
            # model definition file
            pymc_model_file = "./models/pymc_models_iris.py"
            # we have beforehand the following discrete variables
            discrete_variables = ['species']
        # create a model for different algorithms and scores
        algorithms = ["tabu", "hc", "gs", "iamb", "fast.iamb", "inter.iamb"]
        scores = ["loglik-cg", "bic-cg"]
        could_not_fit = 1
        iteration = 0
        number_of_fitted = 0
        while could_not_fit != 0 and iteration < 10:
            # create a new pymc file
            if data_set == "allbus":
                generate_new_pymc_file_allbus(pymc_model_file, result_file, continues_data_file=continues_data_file)
            elif data_set == "iris":
                generate_new_pymc_file_iris(pymc_model_file, result_file)
            could_not_fit = 0
            iteration += 1
            number_of_fitted = 0
            for algo in algorithms:
                if algo in ["tabu", "hc"]:
                    scores = ["bic-cg", "aic-cg"]
                else:
                    scores = [""]
                for score in scores:
                    model_name = f"{data_set}_{algo}{score}".replace("-", "").replace(".", "")
                    ppl_model = PPLModel(model_name, data_file_all_numeric, discrete_variables=discrete_variables,
                                         verbose=False, algo=algo, score=score)
                    # the function returns 1 if the model could not be translated
                    error = ppl_model.generate_pymc(model_name=model_name, save=True, output_file=pymc_model_file, cll=f"cll_{data_set}")
                    if error:
                        could_not_fit += error
                    else:
                        number_of_fitted += 1
        print(f"Could fit: {number_of_fitted} of {could_not_fit + number_of_fitted} bnlearn models")
        # import the beforehand created file
        if data_set == "allbus":
            import scripts.experiments.models.pymc_models_allbus as pymc_allbus_models
            # read all model functions
            functions = [o[1] for o in getmembers(pymc_allbus_models) if isfunction(o[1]) and o[0].startswith("create")]
            create_functions = functions
            # add the hand tuned model
            from scripts.experiments.models.pymc_model_hand_tuned import create_hand_tuned_model
            create_functions.append(create_hand_tuned_model)
        elif data_set == "iris":
            import scripts.experiments.models.pymc_models_iris as pymc_iris_models
            functions = [o[1] for o in getmembers(pymc_iris_models) if isfunction(o[1]) and o[0].startswith("create")]
            create_functions = functions
        # fit all the files
        for func in create_functions:
            ready = False
            tried = 0
            try:
                data, m_fitted = func(fit=True)
            except SamplingError as se:
                # the hand tuned model raisses and error, after 20 tries skip this model
                iteration = 0
                print("Try fit the hand tuned model.")
                number_of_hand_tuned_fits = 5
                while iteration < number_of_hand_tuned_fits:
                    try:
                        print(f"Try: {iteration+1} of {number_of_hand_tuned_fits}")
                        data, m_fitted = func(fit=True)
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

    if fit_sklearn and data_set == "allbus":
        print("Learn and try different sklearn models")
        def mae_score(y_true, y_predict):
            return np.sum(np.abs(y_true - y_predict)) / len(y_true)
        # calculates the accuracy score for variable
        def _test_var(train, test, variable):
            y_train = train[variable]
            del train[variable]
            X_train = train
            y_test = test[variable]
            del test[variable]
            X_test = test
            model_scores = {}
            for model in [model1, model2, model3, model4, model5]:
                m = model()
                m.fit(X_train, y_train)
                acc = accuracy_score(y_test, m.predict(X_test))
                mae = mae_score(y_test, m.predict(X_test))
                model_scores[f"{m.__class__.__name__}"] = {'acc': acc, 'mae': mae}
            return model_scores
        # calculate the scores and save them
        train_data = allbus.train(discretize_all=True)
        test_data = allbus.test(discretize_all=True)
        scores_happy = _test_var(train_data, test_data, "happiness")
        scores_sex = _test_var(train_data, test_data, "sex")
        scores_eastwest = _test_var(train_data, test_data, "eastwest")
        scores_lived_abroad = _test_var(train_data, test_data, "lived_abroad")
        with open(result_file, "a+") as f:
            for model in scores_happy.keys():
                f.write(f"{model}, {scores_happy[model]['acc']}, {scores_happy[model]['mae']}, "
                        f"{scores_sex[model]['acc']}, {scores_sex[model]['mae']}\n")

    print(f"Calculated all scores and fitted all models in {time() - start}s")
    if data_set == "allbus":
        print("\nRESULTS:")
        print(get_results_from_file(result_file))
        generate_happiness_plots(continues_data_file, output_path=os.path.dirname(__file__))
        generate_happiness_plots(continues_data_file, output_path=os.path.dirname(__file__), one_in_all=True)
