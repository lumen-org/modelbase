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

    def create_map_and_clean_data(self, ending_comma=False, index_column=False, whitelist_continuous_variables=None,
                                  whitelist_discrete_variables=None):
        # creates the data map and a new file *_cleaned with just numbers
        categorical_vars = self._get_categorical_vars()
        for discrete_var in whitelist_discrete_variables:
            if discrete_var not in categorical_vars:
                categorical_vars.append(discrete_var)
        for var in whitelist_continuous_variables:
            if var in categorical_vars:
                categorical_vars.remove(var)
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

    def learn_model(self, modelname, whitelist_continuous_variables=[], whitelist_discrete_variables=[],
                    whitelist_edges=[], blacklist_edges=[],
                    simplify=False, simplify_tolerance=0.001, relearn=True, verbose=False):
        # whitelist_edges = [('sex', 'educ'), ('age', 'income'), ('educ', 'income')]
        file = self.csv_data_file[:-4] + "_cleaned.csv"

        gm = GeneratePyMc3Model(file, self.categorical_vars)

        # whitelist_edges = whitelist_edges + [(i,j) for (j,i) in whitelist_edges]

        # adds both edges to the blacklist
        # remove this if you do not want this
        blacklist_edges = blacklist_edges + [(i, j) for (j, i) in blacklist_edges]

        function = gm.generate_model_code(modelname, file=file, fit=True,
                                          continuous_variables=whitelist_continuous_variables,
                                          whitelist=whitelist_edges, blacklist=blacklist_edges,
                                          discrete_variables=self.categorical_vars, simplify=simplify,
                                          simplify_tolerance=simplify_tolerance, relearn=relearn,
                                          verbose=verbose)

        gm.generate_model("../models_video", function, self.data_map, pp_graph=gm.get_description())
        self.generated_model = gm

    def get_description(self):
        return self.generated_model.get_description()

    def get_number_of_parameter(self):
        return self.generated_model.get_number_of_parameter()

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
    whitelist_discrete_variables = []
    whitelist_edges = [('age', 'fare'), ('survived', 'age')]
    #whitelist_edges = []
    blacklist_edges = []
    model = "titanic"

    file = '../data/titanic.csv'
    pymc_model = PyMCModel(file, var_tolerance=0.1)
    pymc_model.create_map_and_clean_data(index_column=False,
                                         whitelist_continuous_variables=whitelist_continuous_variables,
                                         whitelist_discrete_variables=whitelist_discrete_variables)
    pymc_model.learn_model("titanic_4",
                           whitelist_continuous_variables=whitelist_continuous_variables,
                           whitelist_discrete_variables=whitelist_discrete_variables,
                           whitelist_edges=whitelist_edges,
                           blacklist_edges=blacklist_edges, simplify=True, simplify_tolerance=0.3, relearn=True,
                           verbose=True)
    print(pymc_model.get_description())
    print(f"Number of edges {len(pymc_model.get_description()['edges'])}")
    print(f"Learned a model with {pymc_model.get_number_of_parameter()} parameters.")
    # pymc_model.save_graph(file_name="../graph/graph.png", view=True)
