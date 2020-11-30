from time import time
import os
from inspect import getmembers, isfunction

from pymc3.exceptions import SamplingError

from mb_modelbase import EmpiricalModel
from mb_modelbase.models_core.mspnmodel import MSPNModel
from mb_modelbase.models_core.spnmodel import SPNModel
from mb_modelbase.models_core.spflow import SPNModel as SPFlowSPNModel
from mb_modelbase.utils import fit_models, save_models
from mb_modelbase.utils.data_type_mapper import DataTypeMapper

import bin.experiments.cars as cars

from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=RuntimeWarning)
simplefilter(action="ignore", category=UserWarning)


# directory where the generated models for lumen are saved
fitted_models_directory = "./fitted_models/"

sample_size = 45000

# if you do not want to refit models, set the flag to False

fit_bnlearn = False
fit_spn = True
fit_sklearn = False
fit_hand_tuned = True
bnlearn_iris = False

allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                       'lived_abroad': {'No': 0, 'Yes': 1}, 'educ': {'e1': 1, 'e2': 2, 'e3': 3, 'e4': 4, 'e5': 5}}

allbus_backward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                      'lived_abroad': {0: 'No', 1: 'Yes'}, 'educ': {1: 'e1', 2: 'e2', 3: 'e3', 4: 'e4', 5: 'e5'}}

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)

# defined models
spn_models = {
    "test": {
        'emp_cars': lambda: ({'class': EmpiricalModel, 'data': cars.validate(),
                                'fitopts': {'empirical_model_name': 'emp_cars'}}),

        'spflow_cars': lambda: ({'class': SPFlowSPNModel, 'data': cars.train(),
                                       'classopts': {'spn_type': 'spn', 'data_mapping': cars.get_dtm()},
                                       'fitopts': {'var_types': cars.get_spn_types(),
                                                    'cols': 'rdc', 'rows': 'rdc', 'min_instances_slice': 12,
                                                   'min_features_slice': 1, 'multivariate_leaf': False,
                                                   'cluster_univariate': False, 'threshold': 0.8,
                                                   'empirical_model_name': 'emp_cars'}}),

        'spflow_ht_cars': lambda: ({'class': SPFlowSPNModel, 'data': cars.train(),
                                 'classopts': {'spn_type': 'spn', 'data_mapping': cars.get_dtm()},
                                 'fitopts': {'var_types': cars.get_spn_types(),
                                             'cols': 'rdc', 'rows': 'rdc', 'min_instances_slice': 17,
                                             'min_features_slice': 1, 'multivariate_leaf': False,
                                             'cluster_univariate': False, 'threshold': 0.8,
                                             'empirical_model_name': 'emp_cars'}}),

        'spflow_ht_cars_2': lambda: ({'class': SPFlowSPNModel, 'data': cars.train(),
                                    'classopts': {'spn_type': 'spn', 'data_mapping': cars.get_dtm()},
                                    'fitopts': {'var_types': cars.get_spn_types(),
                                                'cols': 'rdc', 'rows': 'rdc', 'min_instances_slice': 20,
                                                'min_features_slice': 1, 'multivariate_leaf': False,
                                                'cluster_univariate': False, 'threshold': 0.1,
                                                'empirical_model_name': 'emp_cars'}}),

        'spflow_ht_cars_3': lambda: ({'class': SPFlowSPNModel, 'data': cars.train(),
                                    'classopts': {'spn_type': 'spn', 'data_mapping': cars.get_dtm()},
                                    'fitopts': {'var_types': cars.get_spn_types(),
                                                'cols': 'rdc', 'rows': 'rdc', 'min_instances_slice': 9,
                                                'min_features_slice': 1, 'multivariate_leaf': False,
                                                'cluster_univariate': False, 'threshold': 0.1,
                                                'empirical_model_name': 'emp_cars'}}),

    },

}

if __name__ == "__main__":
    start = time()
    print("Starting experiments...")

    if fit_bnlearn:
        print("Calculate bnlearn models")
        create_functions = []
        # add the hand tuned model
        if fit_hand_tuned:
            from bin.experiments.models.pymc_models_cars import create_bnlearn_cars

            create_functions.append(create_bnlearn_cars)
        # fit all the files
        for index, func in enumerate(create_functions):
            print(f"Calculate bnlearn model {index + 1} of {len(create_functions)}")
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
        # models = fit_models(spn_models["allbus"], verbose=True, include=[model for model in spn_models["allbus"]])


        # models = fit_models(spn_models["allbus"], verbose=True, include=["spflow_allbus_hand_tuned", "spflow_allbus_hand_gamma", "spflow_allbus_hand_gamma_2"])
        # models = fit_models(spn_models["titanic"], verbose=True, include=[model for model in spn_models['titanic']])
        models = fit_models(spn_models["test"], verbose=True, include=[model for model in spn_models['test'] if "cars" in model])
        save_models(models, fitted_models_directory)
