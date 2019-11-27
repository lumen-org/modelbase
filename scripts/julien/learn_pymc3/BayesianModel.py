from scripts.julien.learn_pymc3.Graph import Graph, Node, DiscreteNode, ContinuousNode

from scripts.julien.learn_pymc3.JSONModelCreator import JSONModelCreator
from scripts.julien.learn_pymc3.JSONReader import JSONReader
from scripts.julien.learn_pymc3.ProbParameter import is_similar, merge_nodes

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
        number_of_merged_parameter = 0
        for node in self.get_graph().get_nodes():
            leafs = node.get_parameter().get_prob_graph().get_leafs()
            for leaf_a in leafs:
                for leaf_b in leafs:
                    if leaf_a is not leaf_b:
                        if is_similar(tolerance, leaf_a, leaf_b):
                            number_of_merged_parameter += 1 if leaf_a.is_discrete() else 2
                            merge_nodes(leaf_a, leaf_b, tolerance)
        if verbose:
            print(f"Merged {number_of_merged_parameter} parameter.")
        self.merged_parameter = number_of_merged_parameter

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
        graph_description['whitelist_edges'] = self.whitelist
        graph_description['blacklist_edges'] = self.blacklist
        graph_description['continuous_vars'] = self.continuous_variables
        graph_description['discrete_vars'] = self.discrete_variables
        return graph_description
