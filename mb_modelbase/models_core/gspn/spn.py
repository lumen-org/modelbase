import copy

import numpy as np
from graphviz import Digraph


class Node(object):
    def __init__(self, scope, children=[]):
        self.children = children
        self.scope = scope

    def add_scope(self, scope):
        self.scope = scope

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children


class SumNode(Node):
    def __init__(self, scope, children=[], weights=[]):
        super().__init__(scope, children)
        self.weights = weights

    def add_weight(self, weight):
        self.weights.append(weight)

    def add_child_with_weight(self, child, weight):
        self.weights.append(weight)
        super().add_child(child)

    def get_weights(self):
        return self.weights

    def get_value(self, obs):
        return np.sum([weight * child.get_value(obs) for weight, child in zip(self.weights, self.children)])

    def __repr__(self):
        return f"+"


class ProductNode(Node):
    def __init__(self, scope, children=[]):
        super().__init__(scope, children)

    def get_value(self, obs):
        return np.product([child.get_value(obs) for child in self.children])

    def __repr__(self):
        return f"x"


class LeafNode(Node):
    def __init__(self, scope, children=[], parameter={}):
        # parameter should have the form {value: probability}
        self.parameter = parameter
        super().__init__(scope, children)

    def get_value(self, obs):
        if self.scope not in obs or obs[self.scope] is True:
            return 1.0

    def get_parameter(self):
        return self.parameter

    def __repr__(self):
        representation = f"{self.scope}: "
        representation += str({ident: f"{value:.2f}" for ident, value in self.parameter.items()})
        return representation


class BernoulliNode(LeafNode):
    def learn_parameter(self, data):
        assert len(np.unique(data)) == 2, "Bernoulli Variables should only have two different values."
        for value in np.unique(data):
            self.parameter[value] = np.count_nonzero(data == value) / len(data)

    def get_value(self, obs):
        assert self.scope in obs.keys(), "Variable not in scope and not marginalized."
        value = obs[self.scope]
        assert value in self.parameter.keys(), "Value not available for variable."
        return self.parameter[value]


class CategoricalNode(LeafNode):
    def learn_parameter(self, data):
      try:
        n = len(data)
      except:
        print("curious")
        n = 1.0
      for value in np.unique(data):
          self.parameter[value] = np.count_nonzero(data == value) /n

    def get_value(self, obs):
        if self.scope not in obs.keys():
            return 1.0
        assert self.scope in obs.keys(), "Variable not in scope and not marginalized."
        value = float(obs[self.scope])
        if value not in self.parameter.keys():
            return 0.0
        return self.parameter[value]


class GaussianNode(LeafNode):
    def learn_parameter(self, data):
        try:
          n = len(data)
        except:
          print("curious")
          n = 1.0
        mu = np.sum(data) / n
        sigma = np.sqrt(np.sum(np.square(data - mu)) / (n))
        self.parameter["mu"] = mu
        self.parameter["sigma"] = sigma if sigma != 0.0 else 1e-12

    def get_value(self, obs):
        if self.scope not in obs.keys():
            return 1.0
        assert self.scope in obs.keys(), "Variable not in scope and not marginalized."
        x = obs[self.scope]
        mu = self.parameter["mu"]
        sigma = self.parameter["sigma"]
        f_x = np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        return f_x


class SPN(object):
    def __init__(self):
        self.root = None
        self.normalize = 1.0

    def set_root(self, root):
        self.root = root

    def get_nodes(self):
        worklist = [self.root]
        nodes = []
        while worklist:
            cur_node = worklist.pop(0)
            nodes.append(cur_node)
            for child in cur_node.get_children():
                worklist.append(child)
        return nodes

    def fit(self, data, var_types, learn_algorithm, params={}):
        learn_algorithm.learn(self, data, var_types, params)
        self.simplify()
        self.calculate_normalize_constant()

    def marginalize(self, marg_out):
        pass

    def predict(self, obs):
        return self.root.get_value(obs) / self.normalize

    def condition(self, obs):
        pass

    def simplify(self):
        pass

    def calculate_normalize_constant(self):
        self.normalize = self.predict({})

    def loglik(self, X):
        score = 0.0
        for data in X:
            score += np.log(self.inference({i:j for i, j in enumerate(data)}))
        return score



    def as_graphviz(self, view=False):
        dot = Digraph(comment="SPN")
        # add nodes
        nodes = self.get_nodes()
        node_to_index = dict()
        for index, node in enumerate(nodes):
            node_to_index[node] = index
            dot.node(str(index), str(node))
        # add edges
        for node in nodes:
            if isinstance(node, SumNode):
                # we need the weights
                for child, weight in zip(node.get_children(), node.get_weights()):
                    dot.edge(str(node_to_index[node]), str(node_to_index[child]), label=f"{weight:.4f}")
            else:
                for child in node.get_children():
                    dot.edge(str(node_to_index[node]), str(node_to_index[child]))
        dot.render(view=view)

    def copy(self):
        return copy.deepcopy(self)



if __name__ == "__main__":
    print("Start")
    n = 100
    a = np.random.choice([1, 2], n, p=[0.4, 0.6])
    b = BernoulliNode(2, [], {})
    b.learn_parameter(a)
    a = np.random.choice([1, 2, 3], n, p=[0.3, 0.6, 0.1])
    c = CategoricalNode(1, [], {})
    c.learn_parameter(a)
    a = np.random.rand(n)
    g = GaussianNode(3)
    g.learn_parameter(a)
    print("Gaussian", g.get_value({3: 0.2}))
    print(b.get_parameter())
    print(c.get_parameter())
    print(g.get_parameter())
    print(c.get_value({1: 3, 2: 2}))
    spn = SPN()
