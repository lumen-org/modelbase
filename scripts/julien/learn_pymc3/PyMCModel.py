from scripts.julien.learn_pymc3.DataTransformer import DataTransformer, DataMap
from scripts.julien.learn_pymc3.JSONModelCreator import JSONModelCreator
from scripts.julien.learn_pymc3.GenerateModel import GeneratePyMc3Model

from pandas import read_csv
import numpy as np
import graphviz


class PyMCModel(object):
    def __init__(self, csv_data_file, var_tolerance=0.2):
        # var_tolerance to detect categorical vars
        # example: if we have 100 data points only 20 different values are allowed
        self.csv_data_file = csv_data_file
        self.data_map = None
        self.var_tolerance = var_tolerance
        self.categorical_vars = None
        self.generated_model = None

    def create_map_and_clean_data(self, ending_comma=False, index_column=False):
        # creates the data map and a new file *_cleaned with just numbers
        categorical_vars = self._get_categorical_vars()
        self.categorical_vars = categorical_vars
        dt = DataTransformer()
        dt.transform(self.csv_data_file, ending_comma=ending_comma,
                     discrete_variables=categorical_vars, index_column=index_column)
        self.data_map = dt.get_map()

    def _get_categorical_vars(self):
        categorical_vars = []
        data = read_csv(self.csv_data_file)
        for name, column in zip(data.columns, np.array(data).T):
            if any([not isinstance(i, float) for i in column]):
                if any([isinstance(i, str) for i in column]):
                    categorical_vars.append(name)
                elif len(np.unique(column)) < len(column) / (1 / self.var_tolerance):
                    categorical_vars.append(name)
        return categorical_vars

    def learn_model(self, modelname, whitelist_continuous_variables=[], whitelist_edges=[], blacklist_edges=[],
                    verbose=False):
        # whitelist_edges = [('sex', 'educ'), ('age', 'income'), ('educ', 'income')]
        file = self.csv_data_file[:-4] + "_cleaned.csv"

        gm = GeneratePyMc3Model(file, self.categorical_vars)
        # pymc3_code = gm.generate_code(whitelist_continuous_variables, whitelist_edges, blacklist_edges)

        function = gm.generate_model_code(modelname, file=file, fit=True,
                                          continuous_variables=whitelist_continuous_variables,
                                          whitelist=whitelist_edges, blacklist=blacklist_edges,
                                          discrete_variables=self.categorical_vars,
                                          verbose=verbose)

        gm.generate_model("../models", function, self.data_map)
        self.generated_model = gm

    def save_graph(self, file_name, view=False):
        descr = self.generated_model.get_description()
        print(descr)
        g = graphviz.Digraph()
        nodes = descr.get('nodes')
        edges = descr.get('edges')
        node_to_number = dict()
        for index, node in enumerate(nodes):
            g.node(str(index), node)
            node_to_number[node] = index
        for node_from, node_to in edges:
            g.edge(str(node_to_number[node_from]), str(node_to_number[node_to]))
        g.render(file_name, view=view)



if __name__ == "__main__":

    whitelist_continuous_variables = []
    whitelist_edges = [('sex', 'educ')]
    blacklist_edges = []
    file = '/home/julien/PycharmProjects/lumen/modelbase/scripts/julien/data/allbus.csv'
    #file = '/home/julien/PycharmProjects/lumen/modelbase/scripts/julien/data/allbus_cleaned.csv'


    pymc_model = PyMCModel(file, var_tolerance=0.1)
    pymc_model.create_map_and_clean_data(index_column=True)
    pymc_model.learn_model("test_jp2",
                           whitelist_continuous_variables=whitelist_continuous_variables,
                           whitelist_edges=whitelist_edges, verbose=False)
    pymc_model.save_graph(file_name="../graph/graph.png", view=True)
    """

    from scripts.julien.learn_pymc3.ProbParameter import generate_prob_graphs, is_similar, merge_nodes, print_prob_table

    from scripts.julien.learn_pymc3.BayesianModel import BayesianModel

    bayesian_model = BayesianModel(continuous_variables=whitelist_continuous_variables,
                                   whitelist=whitelist_edges,
                                   blacklist=blacklist_edges)

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
                            print(leaf_a, leaf_b)
                            #merge_nodes(leaf_a, leaf_b)
                            found = True
                            break
                if found:
                    break

"""