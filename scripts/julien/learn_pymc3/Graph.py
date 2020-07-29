from graphviz import Digraph
import networkx as nx

from scripts.julien.learn_pymc3 import ProbParameter

class Graph(object):
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def get_node(self, node):
        for tmp_node in self.nodes:
            if tmp_node.get_name() == node:
                return tmp_node

    def has_node(self, node):
        if node in self.nodes:
            return True
        else:
            return False

    def get_nodes(self):
        return self.nodes.copy()

    def get_edges(self):
        return self.edges

    def add_edge(self, from_node: str, to_node: str):
        self.edges.append((from_node, to_node))
        self.get_node(from_node).add_child(self.get_node(to_node))
        self.get_node(to_node).add_parent(self.get_node(from_node))

    def export_as_graphviz(self, filename, view=True):
        dot = Digraph()
        node_to_number = dict((node.get_name(), str(index)) for index, node in enumerate(self.nodes))
        for node, index in node_to_number.items():
            dot.node(index, node)
        for node, index in node_to_number.items():
            for child in self.get_node(node).get_children():
                dot.edge(index, node_to_number[child.get_name()])
        dot.render(filename, view=view)

    def get_networkx_object(self):
        networkx_graph = nx.DiGraph()
        node_to_number = dict((node.get_name(), str(index)) for index, node in enumerate(self.nodes))
        for node in self.get_nodes():
            networkx_graph.add_node(node.get_name())
        for node in self.get_nodes():
            for child in self.get_node(node.get_name()).get_children():
                networkx_graph.add_edge(node.get_name(), child.get_name())
        return networkx_graph


class Node(object):
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []
        self.parameter = None

    def add_child(self, node):
        if node not in self.children:
            self.children.append(node)

    def add_parent(self, node):
        if node not in self.parents:
            self.parents.append(node)

    def get_name(self):
        return self.name

    def get_parents(self):
        return self.parents

    def get_parameter(self):
        return self.parameter

    def get_children(self):
        return self.children

    def has_children(self):
        return len(self.children) > 0

    def set_parameter(self, parameter):
        self.parameter = parameter

    def get_discrete_parents(self):
        discrete_parents = []
        for parent in self.parents:
            if parent.is_discrete():
                discrete_parents.append(parent)
        return discrete_parents

    def is_discrete(self):
        raise NotImplementedError("Implemented in DiscreteNode oder ContinuousNode.")


class DiscreteNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.level = None
        self.parameter = None

    def set_level(self, level):
        self.level = level

    def get_level(self):
        return self.level

    def is_discrete(self):
        return True


class ContinuousNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.parameter = None

    def is_discrete(self):
        return False