import copy as cp

import numpy as np
import pandas as pd

from mb_modelbase.utils import get_columns_by_dtype, to_category_cols

from mb_modelbase.models_core import Model

from mb_modelbase.models_core.gspn.spn import SPN
from mb_modelbase.models_core.gspn.glearn import KMeansLearnSPN
from mb_modelbase.models_core.gspn.util import *



class GSPNModel(Model):
  def __init__(self, name, min_number_of_samples=10, max_cluster=8, feature_types={}):
    super().__init__(name)
    self.min_samples = min_number_of_samples
    self.max_cluster = max_cluster
    self.col_to_category = {}
    self.name_to_index = {}
    self.data = None
    # index to feature type (Category, Gaussian)
    self.feature_types = feature_types
    self.gspn = SPN()
    self.known_index_values = dict()
    
  def _set_data(self, df, drop_silently, **kwargs):
    _, cat_names, num_names = get_columns_by_dtype(df)
    self._set_data_mixed(df, drop_silently, num_names, cat_names)
    # save column names
    self.name_to_index = {name: index for index, name in enumerate(df.columns)}
    # make data frame numeric
    key_value, data = as_numeric(df)
    # save values
    self.col_to_category = key_value
    # generate missing feature types
    for index, name in enumerate(df.columns):
      if not index in self.feature_types:
        if name in self.col_to_category:
          self.feature_types[index] = "Category"
        else:
          self.feature_types[index] = "Gaussian"
    # save data
    self.data = data
    # fill index data structure
    self.known_index_values = dict()
    for index, field in enumerate(self.fields):
      self.known_index_values[index] = None
    return []
    
  def _fit(self, **kwargs):
    self.gspn.fit(self.data, self.feature_types, KMeansLearnSPN(self.min_samples, self.max_cluster))
    
  def _marginalizeout(self, keep, remove):
    tmp = {}
    for rem in remove:
      if rem in self.name_to_index:
        self.known_index_values[self.name_to_index[rem]] = True
    return []
    
  def _conditionout(self, keep, remove):
    inverse_name_to_index = {index: name for name, index in self.name_to_index.items()}
    indices = [self._name_to_index[name] for name in remove]
    values = self._condition_values(remove)
    for index, value in zip(indices, values):
      if index in inverse_name_to_index:
        #TODO: check if this is correct
        self.known_index_values[index] = self.col_to_category[inverse_name_to_index[index]][value]
      else:
        self.known_index_values[index] = value
    return []
         
  def _density(self, x):
    inverse_name_to_index = {index: name for name, index in self.name_to_index.items()}
    tmp = self.known_index_values.copy()
    running_index = 0
    for index, key in enumerate(tmp.keys()):
      if tmp[key] is None:
        if inverse_name_to_index[key] is not None and type(x[running_index]) is str:
          column_name = inverse_name_to_index[key]
          tmp[key] = float(self.col_to_category[colname][x[running_index]])
        else:
          tmp[key] = float(x[running_index])
        running_index += 1
    if len(x) != running_index:
      raise Exception("To many values.")
    return self.gspn.predict(tmp)
         
  def _sample(self):
    raise NotImplementedError("Will do this later")
    
  def copy(self, name=None):
    gspncopy = super()._defaultcopy(name)
    gspncopy.min_samples = self.min_samples
    gspncopy.max_cluster = self.max_cluster
    gspncopy.col_to_category = self.col_to_category.copy()
    gspncopy.data = self.data.copy()
    gspncopy.feature_types = self.feature_types.copy()
    gspncopy.gspn = self.gspn
    gspncopy.known_index_values = self.known_index_values.copy()
    gspncopy.name_to_index = self.name_to_index.copy()
    return gspncopy
    

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import pandas as pd
    
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data['feature_names']
    feature_names.append('clasz')
    data_con = pd.DataFrame(data=np.concatenate([X, np.transpose([y])], axis=1), columns=feature_names)
    
    gspn = GSPNModel("iris", feature_types={4:"Category"})
    gspn.set_data(data_con)
    print(gspn.name_to_index)











