from scripts.julien.learn_pymc3.BlogCreator import BlogCreator
from scripts.julien.learn_pymc3.PyMCCreator import PyMCCreator
from scripts.julien.learn_pymc3.BayesianModel import BayesianModel
from scripts.julien.learn_pymc3.ProbParameter import generate_prob_graphs, is_similar, merge_nodes, \
            print_prob_table


class PPLModel():
    def __init__(self, model_name, csv_file, continuous_variables=[], discrete_variables=[], whitelist_edges=[], blacklist_edges=[],
                 score="", algo="", verbose=False):
        self.file = csv_file
        self.model_name = model_name
        self.cont_white = continuous_variables
        self.disc_white = discrete_variables
        self.edges_white = whitelist_edges
        self.edges_black = blacklist_edges
        self.score = score
        self.algo = algo
        self.verbose = verbose

        # clean data
        """
        from bayesian_network_learning.util.DataTransformer import DataTransformer
        dt = DataTransformer()
        dt.transform(file, ending_comma=False, discrete_variables=['cloudy','rain','sprinkler','grass_wet'])
        """
        bayesian_model = BayesianModel(continuous_variables=self.cont_white, discrete_variables=self.disc_white,
                                       whitelist=self.edges_white,
                                       blacklist=self.edges_black, score=self.score, algo=self.algo)
        self.error = bayesian_model.learn_through_r(self.file, relearn=True, verbose=verbose)
        if not self.error:
            descr = bayesian_model.get_graph_description()
            if verbose:
                print("Network Description", descr)
                bayesian_model.get_graph().export_as_graphviz(model_name, view=verbose)

            self.bayesian_model = bayesian_model

            if verbose:
                generate_prob_graphs(bayesian_model)

    def generate_pymc(self, model_name="", output_file="", save=False, continues_data_file=None, cll=None):
        if self.error:
            if self.verbose:
                print(f"Could not fit since bayesian model is wrong (R Error).")
            return 1
        pc = PyMCCreator(self.bayesian_model)
        code = pc.generate()
        parameter = self.bayesian_model.get_number_of_parameter()
        if self.verbose:
            print(f"Learned bayesian network learned with {parameter} parameter.")
        if save:
            code = self.save_complete_pymc_code(code=code, parameter=parameter, model_name=model_name, output_file=output_file, cll=cll)
        return code

    def save_complete_pymc_code(self, code, parameter, model_name, output_file, cll):
        # remove with model and imports
        code = code.replace("import pymc3 as pm\nimport theano.tensor as tt\nwith pm.Model() as model:\n", "")
        # add two tabs before each line of code
        new_code = ""
        for line in code.split("\n"):
            new_code += "        " + line + "\n"
        # remove ending lines
        code = new_code.replace("    \n", "")
        parameter_dict = {
            "code": code.replace("import pymc3 as pm\nimport theano.tensor as tt\nwith pm.Model() as model:\n", ""),
            "parameter": parameter,
            "function_name": "create_" + model_name,
            "model_name": model_name,
            "cll": cll,
        }
        complete_code = """#####################
# {parameter} parameter
#####################
def {function_name}(filename="", modelname="{model_name}", fit=True):
    if fit:
        modelname = modelname
    model = pm.Model()
    with model:
{code}
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, nr_of_posterior_samples=sample_size)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(train_data, auto_extend=False)
        {cll}(m, test_data, model_file, continues_data_file)
    return df, m""".format(**parameter_dict)
        if "NaN" in complete_code:
            if self.verbose:
                print(f"Could not fit because of NAN (Number of parameter: {parameter})")
            return 1
        try:
            f = open(output_file, "a+")
            f.write(complete_code)
            f.write("\n\n")
            f.close()
        except:
            if self.verbose:
                print("Could not save model file.")
        return 0


    def generate_blog(self):
        bc = BlogCreator(self.bayesian_model)
        return bc

def generate_new_pymc_file(file_name, result_file, sample_size=3000, continues_data_file="allbus_happiness_values.dat"):
    code = """#!usr/bin/python
# -*- coding: utf-8 -*-import string

import pymc3 as pm
import theano.tensor as tt

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper
from mb_modelbase.utils.Metrics import *

import scripts.experiments.allbus as allbus_data

# LOAD FILES
test_data = allbus_data.train(discrete_happy=False)
train_data = allbus_data.test(discrete_happy=False)

df = train_data

# SAVE PARAMETER IN THIS FILE
model_file = {model_file}
continues_data_file = {continues_data_file}

sample_size = {sample_size}

allbus_forward_map = {'sex': {'Female': 0, 'Male': 1}, 'eastwest': {'East': 0, 'West': 1},
                       'lived_abroad': {'No': 0, 'Yes': 1}}

allbus_backward_map = {'sex': {0: 'Female', 1: 'Male'}, 'eastwest': {0: 'East', 1: 'West'},
                      'lived_abroad': {0: 'No', 1: 'Yes'}}

dtm = DataTypeMapper()
for name, map_ in allbus_backward_map.items():
    dtm.set_map(forward=allbus_forward_map[name], backward=map_, name=name)

""".replace("{model_file}", f"'{result_file}'").replace("{sample_size}", f"{sample_size}").replace("{continues_data_file}", f"'{continues_data_file}'")
    with open(file_name, "w") as f:
        f.write(code)
