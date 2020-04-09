import copy as cp

import numpy as np
import pandas as pd

from mb_modelbase.models_core import Model
from mb_modelbase.models_core.gspn.spn import SPN

class GSPN(Model):
  def __init__(self, name, **params):
    super().__init__(name)
    self.params = params
    
  def _set_data(self, df, drop_silently, **kwargs):
    pass
    
  def _fit(self):
    pass
    
  def _marginalizeout(self, keep, remove):
    pass
    
  def _conditionout(self, keep, remove):
    pass
         
  def _density(self, x):
    pass
         
  def _sample(self):
    raise NotImplementedError("Will do this later")
    
  def copy(self, name=None):
    pass