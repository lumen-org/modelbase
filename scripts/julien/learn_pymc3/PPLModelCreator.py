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

    def generate_pymc(self, model_name="", output_file="", save=False):
        if self.error:
            print(f"Could not fit since bayesian model is wrong (R Error).")
            return 1
        pc = PyMCCreator(self.bayesian_model)
        code = pc.generate()
        parameter = self.bayesian_model.get_number_of_parameter()
        if self.verbose:
            print(f"Learned bayesian network learned with {parameter} parameter.")
        if save:
            code = self.save_complete_pymc_code(code=code, parameter=parameter, model_name=model_name, output_file=output_file)
        return code

    def save_complete_pymc_code(self, code, parameter, model_name, output_file):
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
        }
        complete_code = """#####################
# {parameter} parameter
#####################
def {function_name}(filename="", modelname="{model_name}", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    model = pm.Model()
    with model:
{code}
    m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m""".format(**parameter_dict)
        if "NaN" in complete_code:
            print(f"Could not fit because of NAN (Number of parameter: {parameter})")
            return 1
        try:
            f = open(output_file, "a+")
            f.write(complete_code)
            f.write("\n\n")
            f.close()
        except:
            print("Could not save model file.")
        return 0


    def generate_blog(self):
        bc = BlogCreator(self.bayesian_model)
        return bc


