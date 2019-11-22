import json
import numpy as np

from scripts.julien.learn_pymc3.Graph import Graph, DiscreteNode, ContinuousNode

from scripts.julien.learn_pymc3.ProbParameter import ProbParameter


class JSONReader(object):
    def __init__(self, bayesian_model):
        self.bayesian_model = bayesian_model

    def parse(self, file: str):
        # load json into variable model
        if not file.endswith('json'):
            raise Exception("JSON file expected.")
        json_file = open(file, "r")
        model_json = json_file.read()
        model = json.loads(model_json)
        # creates a dictionary {node: {index: level}}
        node_to_level = dict({node: dict(enumerate(level)) for node, level in dict(model["levels"]).items()})
        ### build the model ###
        self.bayesian_model.set_level_dict(node_to_level)
        # add nodes to the graph
        graph = Graph()
        for node in model["structure"]["nodes"]:
            # variable is categorical
            if node in node_to_level.keys():
                # node is discrete
                model_node = DiscreteNode(node)
                # remeber the level
                model_node.set_level(node_to_level[node])
            # variable is continual
            else:
                model_node = ContinuousNode(node)
            graph.add_node(model_node)
        # add edges
        for from_node, to_node in model["structure"]["arcs"]:
            graph.add_edge(from_node, to_node)
        # add parameter for each node
        for node in graph.get_nodes():
            parameter = self._calculate_parameter(model["parameter"][node.get_name()], node, self.bayesian_model)
            node.set_parameter(parameter)
        self.bayesian_model.set_graph(graph)
        self.bayesian_model.generate_probability_graphs()
        return self.bayesian_model

    def _calculate_parameter(self, parameter, node, bayesian_model):
        # parameter class
        probability_parameter = ProbParameter(node, bayesian_model)
        # shape of probability tensor have to be calculated
        tensor_shape = []
        # discrete nodes does just have discrete parents
        if node.is_discrete():
            index_dictionary = dict()
            parents = parameter['parents']
            # fill the index dictionary
            index = 0
            for _, parent in enumerate(parents):
                index_dictionary[index] = parent
                index += 1
                tensor_shape.append(bayesian_model.get_level_size(parent))
            index_dictionary[index] = node.get_name()
            tensor_shape.append(bayesian_model.get_level_size(node.get_name()))
            prob_tensor = np.array(parameter['prob'])
            # we want the index of the node at the end
            prob_tensor = np.einsum("i...->...i", prob_tensor)
            assert np.shape(prob_tensor) == tuple(tensor_shape), "Wrong shape detected."
        else:
            # calculate the discrete and continuous parents
            parents = dict(enumerate(parameter['parents']))
            con_parents = self._get_continuous_parents(node)
            # if node has just continuous or no parents
            if len(parents) == len(con_parents) or len(parents) == 0:
                continuous_parents_dict = dict(enumerate(parameter['parents']))
                continuous_parent_indices = list(continuous_parents_dict.keys())
                continuous_parents = list(continuous_parents_dict.values())
                discrete_parents_indices = []
                discrete_parents = []
            # if we have a mixture of continuous and discrete parents
            else:
                continuous_parent_indices = [i-1 for i in parameter['gparents']]
                discrete_parents_indices = [i-1 for i in parameter['dparents']]
                discrete_parents = [node for index, node in parents.items() if index in discrete_parents_indices]
                discrete_parents.reverse()
                continuous_parents = [node for index, node in parents.items() if index in continuous_parent_indices]
            # mu and sigma and factors
            coefficients = parameter['coefficients']
            sd = parameter['sd']
            # one tensor for mu, sd and factors of continous parents
            prob_tensor = np.zeros((len(continuous_parent_indices)+2, len(sd)))
            index_dictionary = {}
            # fill tensor and index_dictionary for discrete parents
            for index, parent in enumerate(discrete_parents):
                tensor_shape.append(bayesian_model.get_level_size(parent))
                index_dictionary[index] = parent
            continuous_index = len(index_dictionary)
            # first two rows of prob_tensor are for mu and sd
            prob_tensor[0] = coefficients[0]
            prob_tensor[1] = sd
            # factors for the other continuous variables
            for index, parent in enumerate(continuous_parents):
                prob_tensor[index+2] = coefficients[index+1]
                index_dictionary[continuous_index+index] = parent
            # add the head current node to the end
            index_dictionary[continuous_index+len(continuous_parents)] = node.get_name()
            # move mu, sd, factors to the end
            prob_tensor = np.einsum("i...->...i", prob_tensor)
            # reshape given tensor shape of categoricals
            prob_tensor = np.reshape(prob_tensor, tuple(tensor_shape + [len(continuous_parent_indices)+2]))
        # set probability tensor and index dictionary
        probability_parameter.set_prob_tensor(prob_tensor)
        probability_parameter.set_index_dict(index_dictionary)
        return probability_parameter

    def _get_discrete_parents(self, node):
        discrete_parents = []
        for parent in node.get_parents():
            if parent.is_discrete():
                discrete_parents.append(parent.get_name())
        return discrete_parents

    def _get_continuous_parents(self, node):
        continuous_parents = []
        for parent in node.get_parents():
            if not parent.is_discrete():
                continuous_parents.append(parent.get_name())
        return continuous_parents


