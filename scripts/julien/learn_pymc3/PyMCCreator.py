import itertools
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from theano.ifelse import ifelse


class PyMCCreator(object):
    def __init__(self, bayesian_model):
        self.bm = bayesian_model
        self.level_dict = self.bm.get_level_dict()

    def generate(self, with_model=True):
        code = ""
        level= ""
        if with_model:
            code = "import pymc3 as pm\n"
            code += "import theano.tensor as tt\n"
            code += "with pm.Model() as model:\n"
            level = ""
        insertion_order = self.bm.get_condition_node_order()

        for node in insertion_order:
            # from bayesian_network_learning.util.ProbParameter import print_prob_table; print_prob_table(node.get_parameter().get_graph(), node.get_name(), True);
            name = node.get_name()
            code += f"{name} = "
            tree = node.get_parameter().get_prob_graph().get_head()

            if node.is_discrete():
                code += "pm.Categorical('{name}', " \
                        "p={prob})\n".format(name=name, prob=self._generate_code_for(node, tree, "prob"))
            else:
                code += "pm.Normal('{name}', " \
                        "mu={mu}, " \
                        "sigma={sigma})\n".format(name=name,
                                                  mu=self._generate_code_for(node, tree, "mu"),
                                                  sigma=self._generate_code_for(node, tree, "sigma"))
        return code

    def _generate_code_for(self, node, tree, case):
        name = node.get_name()
        code = ""
        number_of_children = len(tree.get_children())
        switch = False
        for index, child in enumerate(tree.get_children()):
            # The childs are the leafs
            if child.get_name() == name:
                if case == "prob":
                    code += "[" + ",".join([str(parameter) for parameter in tree.get_parameter()]) + "]"
                    break
                elif case == "mu":
                    code += str(tree.get_parameter(0))
                elif case == "sigma":
                    code += str(tree.get_parameter(1))
            else:
                # discrete node, we have to switch
                if child.is_discrete():
                    if index < number_of_children - 1:
                        code += "tt.switch(tt.eq({name}, " \
                                "{case}), {true}, ".format(name=child.get_name(),
                                                           case=child.get_case(),
                                                           true=self._generate_code_for(node, child, case))
                    else:
                        code += self._generate_code_for(node, child, case)
                    switch = True
                # continuous node, we have a factor
                else:
                    if case == "mu":
                        code += "{name}*{factor}+".format(name=child.get_name(),
                                                          factor=child.get_factor()) \
                                                            + self._generate_code_for(node, child, case)
                    else:
                        assert case == "sigma", "There should be no edge from continuous to discrete nodes."
                        code += self._generate_code_for(node, child, case)

        # insert missing parentheses
        if switch:
            code += ")"*(number_of_children-1)

        return code
