from scripts.julien.learn_pymc3.BayesianModel import BayesianModel
from scripts.julien.learn_pymc3.BlogCreator import BlogCreator
from scripts.julien.learn_pymc3.PyMCCreator import PyMCCreator
from mb_modelbase.models_core.empirical_model import EmpiricalModel

import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import timeit
import os

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model


class GeneratePyMc3Model(object):
    def __init__(self, file, categorical_vars):
        self.file = file
        self.categorical_vars = categorical_vars

    def generate_code(self, continuous_variables, whitelist, blacklist, relearn=True, verbose=True, blog=False):
        """
        :param continuous_variables: list of variables that are set to continuous in the model
        :param whitelist: list of tuples of nodes that will be in the model
        :param blacklist: list of tuples of nodes that will not be in the model
        :param relearn: if the json file is already generated you can skip the learn part
        :param verbose: prints the r-code, modeldescription and generated programs
        :param blog: also prints the blog code
        :return: pymc3_code and description of the model
        """
        bayesian_model = BayesianModel(discrete_variables=self.categorical_vars,
                                       continuous_variables=continuous_variables, whitelist=whitelist,
                                       blacklist=blacklist)
        bayesian_model.learn_through_r(self.file, relearn, verbose)
        descr = bayesian_model.get_graph_description()
        if verbose:
            print(descr)
        if blog:
            bc = BlogCreator(bayesian_model)
            print("#### BLOG ####\n", bc.generate(), sep="")
        pc = PyMCCreator(bayesian_model)
        pymc3_code = pc.generate(with_model=False)
        if verbose:
            print("#### PYMC3 ####\n", pymc3_code, sep="")
            print(f"Learned bayesian network learned with {bayesian_model.get_number_of_parameter()} parameter.")
        return pymc3_code, descr

    def generate_model_code(self, modelname, file, fit, continuous_variables, whitelist, blacklist,
                            discrete_variables=[], relearn=True, verbose=True, blog=False, sample_size=10000,
                            modeldir='models'):
        pymc3_code, descr = self.generate_code(continuous_variables, whitelist, blacklist, relearn, verbose, blog)
        pymc3_code = pymc3_code.replace('\n', '\n                ')
        fun = f"""import os
import pandas as pd
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
def create_fun():
   def code_to_fit(file='{file}', modelname='{modelname}', fit=True):
            # income is gaussian, depends on age
            filepath = os.path.join(os.path.dirname(__file__), '{file}')
            df = pd.read_csv(filepath)
            if fit:
                modelname = modelname + '_fitted'
            # Set up shared variables

            model = pm.Model()
            data = None
            with model:
                {pymc3_code}
            m = ProbabilisticPymc3Model(modelname, model)
            m.nr_of_posterior_samples = {sample_size}
            if fit:
                m.fit(df, auto_extend=False)
            return df, m
   return code_to_fit"""
        f = open('/home/julien/PycharmProjects/lumen/modelbase/scripts/julien/learn_pymc3/ppl_code.py', "w")
        f.write(fun)
        f.close()
        from scripts.julien.learn_pymc3.ppl_code import create_fun
        if verbose:
            print(fun)
        model_function = create_fun()
        # executes the function, we can now use code_to_fit
        return model_function

    def generate_model(self, modeldir, fun):

        # create model and emp model
        mypath = os.path.join(os.path.dirname(__file__), modeldir)
        if not os.path.exists(mypath):
            os.makedirs(mypath)
        start = timeit.default_timer()
        testcasemodel_path = mypath
        testcasedata_path = mypath

        data, m_fitted = fun(fit=True)

        # create empirical model
        name = "emp_" + m_fitted.name
        m_fitted.set_empirical_model_name(name)
        emp_model = EmpiricalModel(name=name)
        emp_model.fit(df=data)

        m_fitted.save(testcasemodel_path)
        emp_model.save(testcasemodel_path)
        if data is not None:
            data.to_csv(os.path.join(testcasedata_path, m_fitted.name + '.csv'), index=False)

        stop = timeit.default_timer()
        print('Time: ', stop - start)


if __name__ == "__main__":
    path = "/home/julien/PycharmProjects/lumen/modelbase/scripts/julien/data/"
    file = 'burglary_cleaned.csv'
    # file = 'students_cleaned.csv'
    # file = 'grade_model.csv'
    # file = 'mixture_gaussian_cleaned.csv'
    file = "allbus_cleaned.csv"
    file = "titanic_cleaned.csv"
    # file = "sprinkler_cleaned.csv"
    file = path + file

    # clean data if necessary
    """
    from bayesian_network_learning.util.DataTransformer import DataTransformer
    dt = DataTransformer()
    dt.transform(file, ending_comma=False, discrete_variables=['cloudy','rain','sprinkler','grass_wet'])
    """

    continuous_variables = []
    whitelist = [('sex', 'age')]
    blacklist = [('income', 'age')]
    continuous_variables = ['income', 'age']
    # whitelist = [('sex', 'educ'), ('age', 'income'), ('educ', 'income')]

    gm = GeneratePyMc3Model(file)
    # pymc3_code = gm.generate_code(continuous_variables, whitelist, blacklist)




    fun_code = gm.generate_model_code("allbus_11", file, True, continuous_variables, whitelist, blacklist, verbose=True)
    exec_namespace = {}

    exec(fun_code, exec_namespace)

    gm.generate_model("../models", code_to_fit)

    # TODO: simplify

    from scripts.julien.learn_pymc3.ProbParameter import generate_prob_graphs, is_similar, merge_nodes, print_prob_table

    """
    bayesian_model = BayesianModel(continuous_variables=continuous_variables, whitelist=whitelist,
                                       blacklist=blacklist)
    bayesian_model.learn_through_r(file, relearn=True, verbose=True)
    if True:
        generate_prob_graphs(bayesian_model)
        conditional_node_order = bayesian_model.get_condition_node_order()
        bayesian_json = {}
        bayesian_json["conditional node order"] = [node.get_name() for node in conditional_node_order]

        for node in conditional_node_order:
            print_prob_table(node.get_parameter().get_prob_graph(), file_name=node.get_name(), view=False, simple=True)
            bayesian_json[node.get_name()] = node.get_parameter().to_json()
        import json
        print(json.dumps(bayesian_json))


        found = False


        for node in bayesian_model.get_graph().get_nodes():
            leafs = node.get_parameter().get_prob_graph().get_leafs()
            for leaf_a in leafs:
                for leaf_b in leafs:
                    if leaf_a is not leaf_b:
                        if is_similar(0.001, leaf_a, leaf_b):
                            merge_nodes(leaf_a, leaf_b)
                            found = True
                            break
                if found:
                    break
    """

