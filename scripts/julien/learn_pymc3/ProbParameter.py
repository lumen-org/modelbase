import itertools
import json

import graphviz
import numpy as np


class ProbParameter(object):
    def __init__(self, node=None, bayesian_model=None):
        self.shape = ()
        self.node = node
        self.bayesian_model = bayesian_model
        self.prob_tensor = None
        self.index_dict = None
        self.prob_graph = None

    def set_prob_tensor(self, prob_tensor):
        self.prob_tensor = prob_tensor

    def get_prob_tensor(self):
        return self.prob_tensor

    def set_index_dict(self, index_dict):
        self.index_dict = index_dict

    def get_index_dict(self):
        return self.index_dict

    def generate_graph(self):
        self.transform_into_tree()

    def get_prob_graph(self):
        return self.prob_graph

    def transform_into_tree(self):
        graph = ProbabilityGraph()
        bayesian_graph = self.bayesian_model.get_graph()
        head_node = ProbabilityNode("HEAD")
        parameter = self.prob_tensor
        graph.set_head(head_node)
        head_node.set_parameter(parameter)
        parents = [head_node]

        for index, variable in self.index_dict.items():
            parents_new = []
            for parent in parents:
                current_parameter = parent.get_parameter()
                children = []
                bayesian_node = bayesian_graph.get_node(variable)
                if bayesian_node.is_discrete():
                    levels = bayesian_node.get_level()
                    for level_index, level in levels.items():
                        child = ProbabilityNode(variable)
                        child.set_case(level)
                        child.set_parameter(current_parameter[level_index])
                        parents_new.append(child)
                        children.append(child)
                        child.set_parent(parent)
                else:
                    child = ProbabilityNode(variable, discrete=False)
                    # we can set a factor
                    if bayesian_node.get_name() != self.node.get_name():
                        factor = current_parameter[2]
                        child.set_factor(factor)
                        child.set_parameter(np.concatenate([current_parameter[0:2], current_parameter[3:]]))
                    else:
                        child.set_parameter(current_parameter)
                    children.append(child)
                    child.set_parent(parent)
                    parents_new.append(child)
                parent.set_children(children)
            parents = parents_new
        self.prob_graph = graph

    def to_json(self):
        graph_dict = {}
        nodes = [self.prob_graph.head]
        index = 0
        node_to_index = {}
        while len(nodes) > 0:
            node = nodes.pop(0)
            node_to_index[node] = index
            node_dict = {}
            node_dict["name"] = node.get_name()
            node_dict["parent"] = node_to_index.get(node.get_parent())
            node_dict["discrete"] = node.is_discrete()
            if not node.is_discrete():
                if len(node.get_children()) > 0:
                    node_dict["factor"] = node.get_factor()
            else:
                node_dict["case"] = node.get_case()
            if len(node.get_children()) == 0:
                node_dict["parameter"] = str(node.get_parameter())
            graph_dict[index] = node_dict
            index += 1
            for child in node.get_children():
                nodes.append(child)
        return graph_dict



class ProbabilityGraph():
    def __init__(self):
        self.head = None

    def add_node(self, node):
        self.nodes.append(node)

    def set_head(self, head):
        self.head = head

    def get_head(self):
        return self.head

    def get_leafs(self):
        leafs = []
        parents = [self.head]
        while len(parents) > 0:
            node = parents.pop(0)
            if len(node.get_children()) == 0:
                leafs.append(node)
            for child in node.get_children():
                parents.append(child)
        return leafs


class ProbabilityNode():
    def __init__(self, name, discrete=True):
        self.name = name
        self.case = None
        self.parameter = None
        self.factor = None
        self.parent = None
        self.children = []
        self.discrete = discrete

    def set_parameter(self, parameter):
        self.parameter = np.nan_to_num(parameter)

    def get_parameter(self, index=None):
        if index is not None and index < len(self.parameter):
            return self.parameter[index]
        else:
            return self.parameter

    def set_factor(self, factor):
        self.factor = factor

    def get_factor(self):
        return self.factor

    def set_case(self, case):
        self.case = case

    def get_case(self):
        return self.case

    def set_children(self, children):
        self.children = children

    def get_children(self):
        return self.children

    def get_name(self):
        return self.name

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def is_discrete(self):
        return self.discrete


def print_prob_table(graph, file_name="graph/test.pdf", view=False, simple=False):
    g = graphviz.Digraph()
    current_node = graph.get_head()
    running_index = 0
    g.node(str(0), current_node.get_name())
    parents = [(running_index, current_node)]
    while len(parents) > 0:
        index, parent = parents.pop(0)
        for child in parent.get_children():
            running_index += 1
            parents.append((running_index, child))
            name = child.get_name()
            case = child.get_case() if child.get_case() else ""
            parameter = ""
            if simple:
                if len(child.get_children()) == 0:
                    parameter = child.get_parameter() if len(np.shape(child.get_parameter())) <= 2 else ""
            else:
                parameter = child.get_parameter() if len(np.shape(child.get_parameter())) <= 2 else ""
            if child.get_factor():
                parameter = child.get_factor()
            node_name = "{name}\n{case}\n{parameter}".format(name=name, case=case, parameter=parameter)
            g.node(str(running_index), node_name)
            g.edge(str(index), str(running_index))
    g.render("graph/" + file_name, view=view)


def generate_prob_graphs(bayesian_model):
    for node in bayesian_model.get_graph().get_nodes():
        print_prob_table(node.get_parameter().get_prob_graph(), node.get_name())


def is_similar(eps, leaf_a, leaf_b):
    assert len(leaf_a.get_children()) == len(leaf_b.get_children()) and len(
        leaf_a.get_children()) == 0, "Nodes should be leafs."
    if leaf_a.is_discrete() and leaf_b.is_discrete():
        if np.abs(leaf_a.get_parameter() - leaf_b.get_parameter()) < eps:
            return True
    if not leaf_a.is_discrete() and not leaf_b.is_discrete():
        mean_a = leaf_a.get_parameter()[0]
        mean_b = leaf_b.get_parameter()[0]
        sd_a = leaf_a.get_parameter()[1]
        sd_b = leaf_b.get_parameter()[1]
        if min(mean_a, mean_b) / (max(mean_a, mean_b)+1e-17) < eps:
            if min(sd_a, sd_b) /  (max(sd_a, sd_b)+1e-17) < eps:
                return True
    return False

def merge_nodes(node_a, node_b, tolerance):
    param_a = node_a.get_parameter()
    param_b = node_b.get_parameter()
    assert node_a.is_discrete() == node_b.is_discrete(), "Both nodes should have the same type."
    if node_a.is_discrete():
        param_both = (param_a+param_b) / 2
        node_a.set_parameter(param_both)
        node_b.set_parameter(param_both)
    else:
        mean_both = (param_a[0]+param_b[0]) / 2
        sd_both = (param_a[1]+param_b[1]) / 2
        param_both = [mean_both, sd_both]
        node_a.set_parameter(param_both)
        node_b.set_parameter(param_both)



