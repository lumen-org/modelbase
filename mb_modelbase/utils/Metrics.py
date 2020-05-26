import random
import os
import re
from subprocess import call

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from mb_modelbase.models_core.base import Condition

def cll_iris(model, test_data, model_file, happy_query_file, income_query_file):
    pass

def cll_allbus(model, test_data, model_file, happy_query_file, income_query_file, overwrite=False):
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
<<<<<<< HEAD

    if overwrite or not os.path.exists(model_file+".csv"):
        with open(model_file+".csv", "w") as f:
            f.write("model_name, acc_happy, mae_happy, acc_sex, mae_sex, acc_ew, mae_ew, acc_la, mae_la, q1, q2, q3, q4\n")
            json = f.write(f"{model_name}, {acc_happy}, {mae_happy}, {acc_sex}, {mae_sex}, {acc_ew}, {mae_ew}, {acc_la}, {mae_la}, {q1}, {q2}, {q3}, {q4}\n")
    else:
        with open(model_file+".csv", "a+") as f:
            json = f.write(f"{model_name}, {acc_happy}, {mae_happy}, {acc_sex}, {mae_sex}, {acc_ew}, {mae_ew}, {acc_la}, {mae_la}, {q1}, {q2}, {q3}, {q4}\n")
    

    density_of_variable_for_same_query(model, test_data, "happiness",
                                       {"sex": "Female", "lived_abroad": "No", "eastwest": "West"},
                                       happy_query_file, stepwidth=0.1)
    density_of_variable_for_same_query(model, test_data, "income",
                                       {"sex": "Male", "lived_abroad": "No", "eastwest": "East"},
                                       income_query_file, stepwidth=100)

    with open(model_file+"_pretty.txt", "w") as f:
        f.write(str(get_results_from_file(model_file+".csv")))

def make_result_pdf(model_file):
    latex_table = str(_get_metric_as_latex_table_entry(model_file+".csv"))
    print (latex_table)
    replace_dict = {"%%%TABLE%%%" : latex_table}
    replace_regex = re.compile("|".join(map(re.escape, replace_dict.keys(  ))))

    templatefile = open('tex_template.tex')
    with open(model_file+"_pretty.tex", "w") as texout:
        for line in templatefile:
            texout.write(replace_regex.sub(lambda match: replace_dict[match.group(0)], line))

    templatefile.close()

    if not os.path.exists("pdf"):
        os.makedirs("pdf/")
    
    os.chdir("pdf/")
    call(["pdflatex", "-output-directory=.", model_file+"_pretty.tex"])

def density_of_variable_for_same_query(model, test_data, query_var, condition_vars, query_file, stepwidth=0.1):
    if str(model.name).startswith("allbus"):
        if query_var == "happiness":
            print(f"Calculated this score for {query_var} on {len([x for x in model.samples.values if x[0] == 'Female' and x[1] == 'West' and x[2] == 'No'])} points.")
        else:
            print(
                f"Calculated this score for {query_var } on {len([x for x in model.samples.values if x[0] == 'Male' and x[1] == 'East' and x[2] == 'No'])} points.")
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
    for value in np.arange(np.min(test_data[query_var]), np.max(test_data[query_var]), stepwidth):
        evidence[query_var] = value
        density = cur_model.density(evidence)
        prediction[value] = density
    # save predictions for query
    with open(query_file, "a+") as f:
        f.write(f"{model.name}, {','.join([str(i) for i in prediction.values()])}\n")

def classify(model, test_data, variable, variable_map=None):
    variable_values = np.unique(test_data[variable].values)
    inference_columns = list(test_data.columns)
    inference_columns.remove(variable)
    evidence = {}
    acc_score = 0
    mae_score = 0
    number_of_samples = len(test_data.values)
    cur_model = model.copy()
    cur_model.marginalize(keep=list(test_data.columns))
    for x, true_value in zip(test_data[inference_columns].values, test_data[variable]):
        # collect variable, value pairs
        for name, value in zip(inference_columns, x):
            evidence[name] = value
        prediction = {}
        # get the prediction value, special case continues variable happiness
        if variable_map is None:
            for value in np.arange(np.min(test_data[variable]), np.max(test_data[variable]), 0.1):
                evidence[variable] = value
                density = cur_model.density(evidence)
                prediction[value] = density
            pred_value = [k for k, v in prediction.items() if v == max(prediction.values())][0]
            # MAPE (https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
            acc_score += np.abs((pred_value+1 - true_value+1)/(true_value+1))
            mae_score += np.abs(pred_value - true_value)
        else:
            # calculate the density value
            for value in variable_values:
                evidence[variable] = value
                prediction[value] = cur_model.density(evidence)
            prediction = [k for k, v in prediction.items() if v == max(prediction.values())][0]
            if prediction == true_value:
                acc_score += 1
            else:
                mae_score += np.abs(variable_map[prediction] - variable_map[true_value])
    return acc_score / number_of_samples, mae_score / number_of_samples

def _query(model, test_data, query_var, condition_vars):
    cur_model = model.copy()
    variables = list(model.names)
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


def _get_metric_as_latex_table_entry(model_file, table_skeleton=True):
    d = pd.read_csv(model_file)
    entries = []
    latex_table = ""
    if table_skeleton:
        header = ["|c" for _ in range(len(d.columns) - 1)]
        header.insert(0, "l|")
        latex_table = """
        \\begin{table}
        \\scriptsize
        \\begin{tabular}{header}
            \\hline
        """.replace("header", "".join(header))#["\\textbf{"+h+"}" for h in header]))

        #latex_table += " & ".join(["\\textbf{"+c+"}".replace("_", "\\_") for c in d.columns]) + "\\\\ \\hline \\hline\n"
        latex_table += " & ".join(["\\textbf{"+c+"}" for c in d.columns]).replace("_", "\\_") + "\\\\ \\hline \\hline\n"
        #latex_table += " & ".join(d.columns).replace("_", "\\_") + "\\\\ \\hline \\hline\n"
    for row in d.iterrows():
        entry = ""
        for i, e in enumerate(row[1]):
            if i > 0:
                if str(float(e)) == "nan":
                    entry += ' & ' + str("N/A")
                else:
                    entry += ' & ' + str(round(float(e), 3))
            else:
                entry += e.replace("_", "-")
        entry += ' \\\\\n\\hline\n'
        latex_table += entry
    if table_skeleton:
        latex_table += """
        	\\end{tabular}
	        \\caption{N/A: not available, -inf: not possible to calculate, due to overflow}
            \\end{table}"""
    return latex_table


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


def generate_happiness_plots(continues_data_file="allbus_happiness_values.dat", output_path="", one_in_all=False):
    d = pd.read_csv(continues_data_file)
    plt.clf()
    if one_in_all:
        fig, axs = plt.subplots(4, 4, figsize=(20,20))
    for index, row in enumerate(d.iterrows()):
        row_dict = dict(row[1])
        model_name = row_dict["model"]
        del row_dict["model"]
        x = np.array([float(i) for i in list(row_dict.keys())])
        y = np.array([float(i) for i in list(row_dict.values())])
        if one_in_all:
            axs[int(index % 4), int(index/4) % 4].plot(x,y)
            axs[int(index % 4), int(index/4) % 4].set_title(model_name)
        else:
            plt.plot(x,y)
            plt.title(model_name)
            plt.savefig(os.path.join(output_path, 'query_happiness_graphs', model_name))
            plt.clf()
    if one_in_all:
        print(os.path.join(output_path, 'query_happiness_graphs', 'merged_graphs'))
        plt.savefig(os.path.join(output_path, 'query_happiness_graphs', 'merged_graphs'))

def generate_income_plots(continues_data_file="allbus_income_values.dat", output_path="", one_in_all=False):
    d = pd.read_csv(continues_data_file)
    plt.clf()
    if one_in_all:
        fig, axs = plt.subplots(4, 4, figsize=(20,20))
    for index, row in enumerate(d.iterrows()):
        row_dict = dict(row[1])
        model_name = row_dict["model"]
        del row_dict["model"]
        x = np.array([float(i) for i in list(row_dict.keys())])
        y = np.array([float(i) for i in list(row_dict.values())])
        if one_in_all:
            axs[int(index % 4), int(index/4) % 4].plot(x,y)
            axs[int(index % 4), int(index/4) % 4].set_title(model_name)
        else:
            plt.plot(x,y)
            plt.title(model_name)
            plt.savefig(os.path.join(output_path, 'query_income_graphs', model_name))
            plt.clf()
    if one_in_all:
        print(os.path.join(output_path, 'query_income_graphs', 'merged_graphs'))
        plt.savefig(os.path.join(output_path, 'query_income_graphs', 'merged_graphs'))

if __name__ == "__main__":
    model_file = "..."
    with open(model_file+"_pretty.txt", "w") as f:
        f.write(str(get_results_from_file(model_file+".csv")))

    make_result_pdf(model_file)