from scripts.julien.learn_pymc3.BayesianModel import BayesianModel
from scripts.julien.learn_pymc3.BlogCreator import BlogCreator
from scripts.julien.learn_pymc3.PyMCCreator import PyMCCreator

from mb_modelbase.models_core.empirical_model import EmpiricalModel
from mb_modelbase.utils.data_type_mapper import DataTypeMapper

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
        self.description = None
        self.bayesian_model = None

    def get_description(self):
        return self.bayesian_model.get_graph_description()

    def get_number_of_parameter(self):
        return self.bayesian_model.get_number_of_parameter()

    def generate_code(self, continuous_variables, whitelist, blacklist, relearn=True, verbose=True, blog=False, simplify=False, simplify_tolerance=0.001):
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
        if simplify:
            bayesian_model.simplify(simplify_tolerance)
        descr = bayesian_model.get_graph_description()
        self.bayesian_model = bayesian_model
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
                            discrete_variables=[], relearn=True, verbose=True, blog=False, sample_size=1000,
                            simplify=False, simplify_tolerance=0.001,
                            modeldir='models'):
        pymc3_code, descr = self.generate_code(continuous_variables, whitelist, blacklist, relearn, verbose, blog, simplify, simplify_tolerance)
        self.description = descr
        pymc3_code = pymc3_code.replace('\n', '\n                ')
        fun = f"""import os
import pandas as pd
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse
from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
def create_fun():
   def code_to_fit(file='{file}', modelname='{modelname}', fit=True, dtm=None, pp_graph=None):
            # income is gaussian, depends on age
            filepath = os.path.join(os.path.dirname(__file__), '{file}')
            df_model_repr = pd.read_csv(filepath)
            df_orig = dtm.backward(df_model_repr, inplace=False)
            if fit:
                modelname = modelname + '_fitted'
            # Set up shared variables

            model = pm.Model()
            data = None
            with model:
                {pymc3_code}
            m = ProbabilisticPymc3Model(modelname, model, data_mapping=dtm, probabilistic_program_graph=pp_graph)
            m.nr_of_posterior_samples = {sample_size}
            if fit:
                m.fit(df_orig, auto_extend=False)
            return df_orig, m
   return code_to_fit"""
        f = open('./ppl_code.py', "w")
        f.write(fun)
        f.close()
        from scripts.julien.learn_pymc3.ppl_code import create_fun
        if verbose:
            print(fun)
        model_function = create_fun()
        # executes the function, we can now use code_to_fit
        return model_function

    def generate_model(self, modeldir, fun, data_map, pp_graph, verbose=False):

        # create model and emp model
        mypath = os.path.join(os.path.dirname(__file__), modeldir)
        if not os.path.exists(mypath):
            os.makedirs(mypath)
        start = timeit.default_timer()
        testcasemodel_path = mypath
        testcasedata_path = mypath

        dtm = DataTypeMapper()
        for name, map_ in data_map.map.items():
            dtm.set_map(forward='auto', backward=map_, name=name)
        if verbose:
            print("pp_graph:", pp_graph)

        data, m_fitted = fun(fit=True, dtm=dtm, pp_graph=pp_graph)

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
