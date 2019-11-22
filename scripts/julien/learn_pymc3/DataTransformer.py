import numpy as np
import pandas as pd


class DataTransformer(object):
    def __init__(self):
        self.datamap = None
        self.data = None

    def transform(self, file_name, ending_comma=False, discrete_variables=[], index_column=False):
        """
        ending_comma = False
            after each row in the csv there is a comma without data
        """
        self.read_csv(file_name, index_column, ending_comma)
        self.datamap = DataMap()
        self.datamap.generate_map(self.data)
        for column in discrete_variables:
            self.data[column] = pd.Series(self.data[column]).map(self.datamap.get_map(column))
        self.save_csv(file_name)

    def read_csv(self, file_name, index_column, ending_comma=False):
        self.data = pd.read_csv(file_name, na_filter=False, index_col=0 if index_column else False)
        if ending_comma:
            self.data = self.data.drop(columns=self.data.columns[len(self.data.columns)-1])

    def save_csv(self, file_name):
        self.data.to_csv(file_name[:-4]+"_cleaned.csv", index=False)

    def get_map(self):
        return self.datamap


class DataMap(object):
    def __init__(self):
        self.map = dict()

    def generate_map(self, data):
        for column in data:
            values = np.unique(data[column])
            values_to_int = {i: value for i, value in enumerate(values)}
            self.map[column] = values_to_int

    def get_map(self, column, reverse=True):
        if reverse:
            return {value: key for key, value in self.map[column].items()}
        else:
            return self.map[column]

    def __repr__(self):
        repr = ""
        for key, value in self.map.items():
            repr += f"{key}: {value}\n"
        return repr

if __name__ == "__main__":
    pass

    file = "/home/julien/PycharmProjects/util/bayesian_network_learning/data/albus.csv"
    dt = DataTransformer()
    dt.transform(file, discrete_variables=["sex","educ","eastwest","happiness","health","lived_abroad","spectrum"], index_column=True)

    print(dt.get_map())
