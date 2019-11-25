from scripts.julien.learn_pymc3.DataTransformer import DataTransformer, DataMap
from scripts.julien.learn_pymc3.JSONModelCreator import JSONModelCreator
from scripts.julien.learn_pymc3.GenerateModel import GeneratePyMc3Model

from pandas import read_csv
import numpy as np


class PyMCModel(object):
    def __init__(self, csv_data_file, var_tolerance=0.2):
        # var_tolerance to detect categorical vars
        # example: if we have 100 data points only 20 different values are allowed
        self.csv_data_file = csv_data_file
        self.data_map = None
        self.var_tolerance = var_tolerance
        self.categorical_vars = None

    def create_map_and_clean_data(self, ending_comma=False):
        # creates the data map and a new file *_cleaned with just numbers
        categorical_vars = self._get_categorical_vars()
        self.categorical_vars = categorical_vars
        dt = DataTransformer()
        dt.transform(self.csv_data_file, ending_comma=ending_comma,
                     discrete_variables=categorical_vars)
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

    def learn_model(self, modelname, whitelist_continuous_variables=[], whitelist_edges=[], blacklist_edges=[]):
        # whitelist_edges = [('sex', 'educ'), ('age', 'income'), ('educ', 'income')]
        file = self.csv_data_file[:-4] + "_cleaned.csv"

        gm = GeneratePyMc3Model(file, self.categorical_vars)
        # pymc3_code = gm.generate_code(whitelist_continuous_variables, whitelist_edges, blacklist_edges)

        function = gm.generate_model_code(modelname, file=file, fit=True, continuous_variables=whitelist_continuous_variables,
                                          whitelist=whitelist_edges, blacklist=blacklist_edges,
                                          discrete_variables=self.categorical_vars,
                                          verbose=False)


        gm.generate_model("/home/luca_ph/Documents/projects/graphical_models/code/models_ppl", function, self.data_map)
        #gm.generate_model("../../../../models_ppl", function, self.data_map)


if __name__ == "__main__":

    whitelist = {
        'categorical': [],
        'quantitative': [],
    }

    pymc_model = PyMCModel("../data/titanic.csv", var_tolerance=0.1)
    pymc_model.create_map_and_clean_data()
    pymc_model.learn_model("test_jp2",
                           whitelist_continuous_variables=['age'],
                           whitelist_edges=[('pclass', 'survived'), ('sex', 'survived')])
