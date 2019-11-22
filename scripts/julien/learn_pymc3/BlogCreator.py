import itertools

class BlogCreator(object):
    def __init__(self, bayesian_model):
        self.bm = bayesian_model
        self.level_dict = self.bm.get_level_dict()

    def generate(self):
        code = ""
        insertion_order = self.bm.get_condition_node_order()
        for var in insertion_order:
            code += self._generate_code_for_variable(var) + "\n"
        return code

    def _generate_code_for_variable(self, var):
        # get parameter
        prob_parameter = var.get_parameter()
        parents = var.get_parents()
        name = var.get_name()
        prob_tensor = prob_parameter.get_prob_tensor()
        index_dict = prob_parameter.get_index_dict()
        # calculate the subset of discrete parents
        discrete_parents = []
        # create cases and indices
        indices = []
        cases = []
        for index, node in index_dict.items():
            if self.bm.get_graph().get_node(node).is_discrete() and node != name:
                discrete_parents.append(node)
                indices.append(list(self._get_level(node).keys()))
                cases.append(list([int(case) for case in self._get_level(node).values()]))
        indices = list(itertools.product(*indices))
        if len(discrete_parents) > 1:
            cases = list(itertools.product(*cases))
        elif len(discrete_parents) == 1:
            cases = [case for case in cases[0]]

        # create code
        code = ""
        # choose distribution type of variable
        if var.is_discrete():
            code += "random Integer {name} ~ ".format(name=name)
        else:
            code += "random Real {name} ~ ".format(name=name)
        # make a table of categorical parents
        if len(indices) > 1:
            remaining_nodes = {key: value for key, value in index_dict.items() if value not in discrete_parents}
            if len(discrete_parents) == 1:
                code += "\n\t case {parents} in ".format(parents=",".join(discrete_parents)) + "{"
            else:
                code += "\n\t case [{parents}] in ".format(parents=",".join(discrete_parents)) + "{"
            for index, case in zip(indices, cases):
                if isinstance(case, tuple):
                    case = "[" + ",".join([str(c) for c in case]) + "]"
                code += "\n\t\t {case} -> {prob},".format(case=case, prob=self._get_distribution(prob_tensor[index], remaining_nodes, var.is_discrete()))
            # we have to remove the last ,
            code = code[:-1] + "\n};"
        else:
            code += self._get_distribution(prob_tensor, index_dict, var.is_discrete()) + ";"

        return code

    def _get_distribution(self, probs, nodes, discrete):
        code = ""
        if discrete:
            # there should only be one index left
            assert len(nodes) == 1, "There should only be one node left."
            levels = self._get_level(list(nodes.values())[0])
            assert len(levels) == len(probs), "For each level should be a prob left."
            parameter_string = ["{case}->{prob}".format(case=case, prob=prob) for case, prob in zip(levels.values(), probs)]
            code = "Categorical({" + ",".join(parameter_string) + "})"
        else:
            # minus one since there is the categorical node left
            assert len(nodes)-1 == len(probs[2:]), "To much nodes or factors."
            factor_term = ""
            for node, factor in zip(nodes.values(), probs[2:]):
                factor_term += "+{factor}*{node}".format(factor=factor, node=node)
            code = "Gaussian({mu}{factor}, {sd})".format(mu=probs[0], sd=probs[1]+10e-5 if probs[1] == 0 else probs[1], factor=factor_term)

        return code

    def _get_level(self, var_name):
        if var_name in self.level_dict.keys():
            return self.level_dict[var_name]
        else:
            return None
