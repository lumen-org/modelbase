import random
import os

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from mb_modelbase.models_core.base import Condition

def cll(model, test_data, model_file):
    model_name = model.name
    # happiness scores
    acc_happy, hit_happy = classify(model, test_data, "happiness", map={'h0': 0, 'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6, 'h7': 7, 'h8': 8, 'h9': 9, 'h10': 10})
    # sex scores
    acc_sex, hit_sex = classify(model, test_data, "sex", map={'Female': 0, 'Male': 1})
    # query scores
    q1 = _query(model, test_data, "sex", ["income", "educ"])
    q2 = _query(model, test_data, "happiness", ["sex", "eastwest", "health"])
    q3 = _query(model, test_data, "eastwest", ["income", "educ", "health"])
    q4 = _query(model, test_data, "lived_abroad", ["income", "educ", "sex", "happiness"])
    with open(model_file, "a+") as f:
        json = f.write(f"{model_name}, {acc_happy}, {hit_happy}, {acc_sex}, {hit_sex}, {q1}, {q2}, {q3}, {q4}\n")

def classify(model, test_data, variable, map):
    variable_values = np.unique(test_data[variable].values)
    inference_columns = list(test_data.columns)
    inference_columns.remove(variable)
    evidence = {}
    acc_score = 0
    hit_score = 0
    number_of_samples = len(test_data.values)
    for x, true_happiness in zip(test_data[inference_columns].values, test_data[variable]):
        for name, value in zip(inference_columns, x):
            evidence[name] = value
        prediction = {}
        for value in variable_values:
            evidence[variable] = value
            prediction[value] = model.density(evidence)
        prediction = [k for k, v in prediction.items() if v == max(prediction.values())][0]
        if prediction == true_happiness:
            acc_score += 1
        if model.__class__.__name__ == "MSPNModel" and variable == "happiness":
            hit_score += np.abs(prediction - true_happiness)
        else:
            hit_score += np.abs(map[prediction] - map[true_happiness])
    return (acc_score / number_of_samples, hit_score / number_of_samples)


def _query(model, test_data, query_var, condition_vars):
    cur_model = model.copy()
    variables = list(test_data.columns)
    marg_variables = variables.copy()
    marg_variables.remove(query_var)
    for var in condition_vars:
        marg_variables.remove(var)
    cur_model.marginalize(remove=marg_variables)
    pll = 0
    query_variable_values = np.unique(test_data[query_var].values)
    for index, data in test_data.iterrows():
        x = {}
        for evidence in condition_vars:
            evidence_value = data[evidence]
            x[evidence] = evidence_value
        for value in query_variable_values:
            x[query_var] = value
            pll += np.log(cur_model.density(x))
    return pll


def _get_metric_as_table_entry(model_file):
    d = pd.read_csv(model_file)
    entries = []
    for row in d.iterrows():
        entry = ""
        for i, e in enumerate(row[1]):
            if i > 0:
                entry += ' & ' + str(round(float(e), 3))
            else:
                entry += e.replace("_", "-")
        entry += '\\\\'
        entries.append(entry)
    return entries

def get_results_from_file(model_file):
    d = pd.read_csv(model_file)
    table = PrettyTable()
    table.field_names = d.columns
    for row in d.iterrows():
        entry = []
        for i, e in enumerate(row[1]):
            if i > 0:
                entry.append(str(round(float(e), 3)))
            else:
                entry.append(e.replace("_", "-"))
        table.add_row(entry)
    return table



if __name__ == "__main__":
    file = "/home/julien/PycharmProjects/modelbase/scripts/julien/allbus.dat"
    entries = _get_metric_as_table_entry(file)
    for e in entries:
        print(e)
    #_print_metric_as_table_entry(os.path.join(os.path.dirname(__file__), "allbus.dat"))
