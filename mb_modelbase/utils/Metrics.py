import random
import os

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from mb_modelbase.models_core.base import Condition


def cll_allbus(model, test_data, model_file, continues_data_file):
    model_name = model.name
    # happiness scores
    acc_happy, mae_happy = classify(model, test_data, "happiness")
    # sex scores
    acc_sex, mae_sex = classify(model, test_data, "sex", variable_map={'Female': 0, 'Male': 1})
    # eastwest scores
    acc_ew, mae_ew = classify(model, test_data, 'eastwest', variable_map={'East': 0, 'West': 1})
    # lived_abroad scores
    acc_la, mae_la = classify(model, test_data, 'lived_abroad', variable_map={'No': 0, 'Yes': 1})
    # query scores
    q1 = _query(model, test_data, "sex", ["income", "educ"])
    q2 = _query(model, test_data, "happiness", ["sex", "eastwest", "health"])
    q3 = _query(model, test_data, "eastwest", ["income", "educ", "health"])
    q4 = _query(model, test_data, "lived_abroad", ["income", "educ", "sex", "happiness"])
    with open(model_file, "a+") as f:
        json = f.write(f"{model_name}, {acc_happy}, {mae_happy}, {acc_sex}, {mae_sex}, {acc_ew}, {mae_ew}, {acc_la}, {mae_la}, {q1}, {q2}, {q3}, {q4}\n")
    density_of_happiness_for_same_query(model, test_data, "happiness",
                                        {"sex": "Female", "educ": 3, "eastwest": "East", "health": 4},
                                        continues_data_file)

def density_of_happiness_for_same_query(model, test_data, query_var, condition_vars, continues_data_file):
    cur_model = model.copy()
    marg_variables = []
    marg_variables.append(query_var)
    for var in condition_vars.keys():
        marg_variables.append(var)
    cur_model.marginalize(keep=marg_variables)
    prediction = {}
    evidence = {}
    for name, value in condition_vars.items():
        evidence[name] = value
    for value in np.arange(np.min(test_data[query_var]), np.max(test_data[query_var]), 0.1):
        evidence[query_var] = value
        density = cur_model.density(evidence)
        prediction[value] = density
    # save predictions for query
    with open(continues_data_file, "a+") as f:
        f.write(f"{model.name}, {','.join([str(i) for i in prediction.values()])}\n")

def classify(model, test_data, variable, variable_map=None):
    variable_values = np.unique(test_data[variable].values)
    inference_columns = list(test_data.columns)
    inference_columns.remove(variable)
    evidence = {}
    acc_score = 0
    mae_score = 0
    number_of_samples = len(test_data.values)
    for x, true_value in zip(test_data[inference_columns].values, test_data[variable]):
        # collect variable, value pairs
        for name, value in zip(inference_columns, x):
            evidence[name] = value
        prediction = {}
        # get the prediction value, special case continues variable happiness
        if variable_map is None:
            for value in np.arange(np.min(test_data[variable]), np.max(test_data[variable]), 0.1):
                evidence[variable] = value
                density = model.density(evidence)
                prediction[value] = density
            pred_value = [k for k, v in prediction.items() if v == max(prediction.values())][0]
            # MAPE (https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
            acc_score += np.abs((pred_value+1 - true_value+1)/(true_value+1))
            mae_score += np.abs(pred_value - true_value)
        else:
            # calculate the density value
            for value in variable_values:
                evidence[variable] = value
                prediction[value] = model.density(evidence)
            prediction = [k for k, v in prediction.items() if v == max(prediction.values())][0]
            if prediction == true_value:
                acc_score += 1
            else:
                mae_score += np.abs(variable_map[prediction] - variable_map[true_value])
    return acc_score / number_of_samples, mae_score / number_of_samples

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
                if str(float(e)) == "nan":
                    entry.append(str("N/A"))
                else:
                    entry.append(str(round(float(e), 3)))
            else:
                entry.append(e.replace("_", "-"))
        table.add_row(entry)
    return table


def generate_happiness_plots(continues_data_file="allbus_happiness_values.dat"):
    d = pd.read_csv(continues_data_file)
    plt.clf()
    for row in d.iterrows():
        row_dict = dict(row[1])
        model_name = row_dict["model"]
        del row_dict["model"]
        x = np.array([float(i) for i in list(row_dict.keys())])
        y = np.array([float(i) for i in list(row_dict.values())])
        plt.plot(x,y)
        plt.title(model_name)
        #raise NotImplementedError("DO PATH!!!!")
        plt.savefig(os.path.join('/home/julien/PycharmProjects/modelbase/scripts/experiments/query_happiness_graphs', model_name))
        plt.clf()
        #plt.savefig(f"{os.path.join('scripts/experiments/query_happiness_graphs', model_name)}.png")



if __name__ == "__main__":
    file = "/home/julien/PycharmProjects/modelbase/scripts/julien/allbus.dat"
    entries = _get_metric_as_table_entry(file)
    for e in entries:
        print(e)
    # _print_metric_as_table_entry(os.path.join(os.path.dirname(__file__), "allbus.dat"))
