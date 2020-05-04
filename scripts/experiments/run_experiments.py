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

from mb_modelbase import EmpiricalModel
from mb_modelbase.models_core.mspnmodel import MSPNModel
from mb_modelbase.models_core.spnmodel import SPNModel
from mb_modelbase.models_core.spflow import SPNModel as SPFlowSPNModel
from mb_modelbase.utils.Metrics import cll, get_results_from_file
from mb_modelbase.utils import fit_models, save_models

import scripts.experiments.allbus as allbus
from scripts.julien.learn_pymc3.PPLModelCreator import PPLModel, generate_new_pymc_file

# file where results from the experiments are saved
result_file = "allbus_results.dat"

# directory where the generated models for lumen are saved
fitted_models_directory = "./fitted_models/"

# defined models
spn_models = {
    # NIPS 2020 Models
    # EMP
    'emp_allbus': lambda: ({'class': EmpiricalModel, 'data': allbus.train(),
                            'fitopts': {'empirical_model_name': 'emp_allbus'}}),
    'emp_allbus_continues': lambda: ({'class': EmpiricalModel, 'data': allbus.train_continuous(),
                                      'fitopts': {'empirical_model_name': 'emp_allbus_continues'}}),
    # SPFLOW
    'spflow_allbus_spn': lambda: ({'class': SPFlowSPNModel, 'data': allbus.train(),
                                   'classopts': {'spn_type': 'spn'},
                                   'fitopts': {'var_types': allbus.spn_parameters["julien"],
                                               'empirical_model_name': 'emp_allbus'}}),
    'spflow_allbus_mspn': lambda: ({'class': SPFlowSPNModel, 'data': allbus.train(),
                                    'classopts': {'spn_type': 'mspn'},
                                    'fitopts': {'var_types': allbus.spn_metatypes["julien"],
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
}

if __name__ == "__main__":
    start = time()
    print("Starting experiments...")

    print("Resetting results")
    with open(result_file, "w") as f:
        f.write(f"modelname, happiness_acc, happiness_hit, sex_acc, sex_hit, q1, q2, q3, q4\n")

    print("Calculate SPN models")
    models = fit_models(spn_models, verbose=True, include=spn_models)
    print("Calculate SPN model scores (skip emp)")
    number_of_spn_models = len([m for m in models if not m.startswith("emp")])
    for index, (model_name, property) in enumerate(models.items()):
        print(f"Calculate SPN model {index} of {number_of_spn_models}")
        if not model_name.startswith("emp"):
            if model_name.startswith("spn_"):
                pass
            else:
                cll(property["model"], allbus.test(discrete_happy=False), result_file)
    save_models(models, fitted_models_directory)

    print("Calculate bnlearn models")
    # Since there are random initialisations, you have to restart the calculation different times, until all have fit
    # BNLearn just works on numbers
    allbus_file_all_discrete = allbus._numeric_data
    # model definition file
    pymc_model_file = "./models/pymc_models_allbus.py"
    # we have beforehand the following discrete variables
    discrete_variables = ['sex', 'eastwest', 'lived_abroad']
    # create a new pymc file
    generate_new_pymc_file(pymc_model_file, result_file)
    # create a model for different algorithms and scores
    algorithms = ["tabu", "hc", "gs", "iamb", "fast.iamb", "inter.iamb"]
    scores = ["loglik-cg", "bic-cg"]
    could_not_fit = 0
    number_of_fitted = 0
    for algo in algorithms:
        if algo in ["tabu", "hc"]:
            scores = ["bic-cg", "aic-cg"]
        else:
            scores = [""]
        for score in scores:
            model_name = f"allbus_{algo}{score}".replace("-", "").replace(".", "")
            ppl_model = PPLModel(model_name, allbus_file_all_discrete, discrete_variables=discrete_variables,
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
    # import scripts.experiments.models.pymc_models_allbus as pymc_allbus_models
    # create_functions.append(create_hand_tuned_model)
    # fit all the files
    for func in create_functions:
        ready = False
        tried = 0
        data, m_fitted = func(fit=True)
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

    print("Learn and try different sklearn models")
    def hit_score(y_true, y_predict):
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
            hit = hit_score(y_test, m.predict(X_test))
            model_scores[f"{m.__class__.__name__}"] = {'acc': acc, 'hit': hit}
        return model_scores
    # calculate the scores and save them
    train_data = allbus.train(discretize_all=True)
    test_data = allbus.test(discretize_all=True)
    scores_happy = _test_var(train_data, test_data, "happiness")
    scores_sex = _test_var(train_data, test_data, "sex")
    with open(result_file, "a+") as f:
        for model in scores_happy.keys():
            f.write(f"{model}, {scores_happy[model]['acc']}, {scores_happy[model]['hit']}, "
                    f"{scores_sex[model]['acc']}, {scores_sex[model]['hit']}\n")

    print(f"Calculated all scores and fitted all models in {time()-start}s")
    print("\nRESULTS:")
    print(get_results_from_file(result_file))