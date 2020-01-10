from scripts.eurovis2020.learn_pymc3.Graph import Graph, Node, DiscreteNode, ContinuousNode

from scripts.eurovis2020.learn_pymc3.JSONModelCreator import JSONModelCreator
from scripts.eurovis2020.learn_pymc3.JSONReader import JSONReader
from scripts.eurovis2020.learn_pymc3.ProbParameter import is_similar, merge_nodes

import numpy as np


class BayesianModel(object):
    def __init__(self, discrete_variables=[], continuous_variables=[], blacklist=None, level_dict=None, whitelist=None):
        self.graph = None
        self.level_dict = level_dict
        self.whitelist = whitelist
        self.blacklist = blacklist
        self.discrete_variables = discrete_variables
        self.continuous_variables = continuous_variables
        self.merged_parameter = 0

    def learn(self, data):
        raise NotImplementedError("Have to be done by using structure and parameter learning.")

    def learn_through_r(self, data_file, relearn=True, verbose=False):
        if relearn:
            jmc = JSONModelCreator(data_file, self.whitelist, self.discrete_variables, self.continuous_variables, self.blacklist)
            (json_file, discrete_vars, continuous_vars) = jmc.generate_model_as_json_with_r(verbose)
            self.discrete_variables = discrete_vars
            self.continuous_variables = continuous_vars
        else:
            json_file = data_file + ".json"
        json_reader = JSONReader(self)
        bayesian_model = json_reader.parse(json_file)

    def simplify(self, tolerance, verbose=False):
        prepared_nodes = []
        most_simplify = 0
        cur_simplify = 0
        name = None
        number_of_merged_parameter = 0
        to_merge = []
        for node in self.get_graph().get_nodes():
            to_merge = []
            leafs = node.get_parameter().get_prob_graph().get_leafs()
            cur_simplify = 0
            for leaf_a in leafs:
                for leaf_b in leafs:
                    if leaf_a is not leaf_b:
                        if is_similar(tolerance, leaf_a, leaf_b):
                            to_merge.append(leaf_a)
                            to_merge.append(leaf_b)
                            #cur_simplify += 1
                            #if leaf_a not in prepared_nodes:
                            #    prepared_nodes.append(leaf_a)
                            #if leaf_b not in prepared_nodes:
                            #    prepared_nodes.append(leaf_b)
                            #merge_nodes(leaf_a, leaf_b, tolerance)
            params = 0
            mu = 0
            sd = 0
            for leaf in to_merge:
                if leaf.is_discrete():
                    params += leaf.get_parameter()
                else:
                    mu += leaf.get_parameter()[0]
                    sd += leaf.get_parameter()[1]
            if len(to_merge) > 0:
                new_param = params/len(to_merge)
                new_mean = mu/len(to_merge)
                new_sd = sd/len(to_merge)
                for leaf in to_merge:
                    if leaf.is_discrete():
                        cur_simplify += 1
                        leaf.set_parameter(new_param)
                    else:
                        cur_simplify += 2
                        leaf.set_parameter([new_mean, new_sd])
        self.merged_parameter = int(cur_simplify)

    def generate_probability_graphs(self):
        for node in self.graph.get_nodes():
            node.get_parameter().generate_graph()

    def get_graph(self):
        return self.graph

    def set_graph(self, graph):
        self.graph = graph

    def set_level_dict(self, level_dict):
        self.level_dict = level_dict

    def add_node(self, node):
        if self.graph:
            self.graph = Graph()
        self.graph.add_node(node)

    def add_edge(self, node_from, node_to):
        if self.graph.has_node(node_from) and self.graph.has_node(node_to):
            self.graph.add_edge(node_from, node_to)
        else:
            raise Exception("Please first insert nodes.")

    def add_level(self, level):
        self.level_dict = level

    def get_level_size(self, node):
        if node in self.level_dict:
            return len(self.level_dict[node])
        else:
            return 0

    def get_level_dict(self):
        return self.level_dict

    def get_level(self, node, index=None):
        if node in self.level_dict:
            if not index:
                return self.level_dict[node]
            else:
                return self.level_dict[node][index]

    def get_condition_node_order(self):
        """
        Computes the insertion order for the blog creation.
        :return: order to insert nodes
        """
        graph = self.get_graph()
        nodes = graph.get_nodes()
        inserted = []
        order = []
        while nodes:
            curr = nodes.pop(0)
            if any([parent not in inserted for parent in curr.get_parents()]):
                nodes.append(curr)
            else:
                inserted.append(curr)
                order.append(curr)
        return order

    def get_number_of_parameter(self):
        parameter_size = 0
        for nodes in self.get_graph().get_nodes():
            parameter_size += np.product(np.shape(nodes.get_parameter().get_prob_tensor()))
        return parameter_size - self.merged_parameter

    def get_graph_description(self):
        nodes = [node.get_name() for node in self.get_graph().get_nodes()]
        edges = self.get_graph().get_edges()
        graph_description = {'nodes': nodes, 'edges': edges}
        graph_description['enforced_edges'] = self.whitelist
        graph_description['forbidden_edges'] = self.blacklist
        enforced_node_dtypes = dict()
        for node in self.continuous_variables:
            enforced_node_dtypes[node] = 'numerical'
        for node in self.discrete_variables:
            enforced_node_dtypes[node] = 'string'
        graph_description['enforced_node_dtypes'] = enforced_node_dtypes
        return graph_description
