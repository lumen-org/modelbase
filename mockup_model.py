# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import logging
import numpy as np
from numpy import pi, exp, matrix, ix_, nan

import utils
import models as md
from models import AggregationTuple, SplitTuple, ConditionTuple
import domains as dm

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class MockupModel(md.Model):
    """A mock up model solely for testing of the abstract Model class.

    You can use it as any other model class, i.e. you can fit data to it, marginalize, aggregate, density-query, ...
    it.

    Fitting: No actual fitting will be done, however, suitable fields for the model are derived from the data. The
     Fields actually match whatever you provide as data, e.g. if you provide data with a categorical column that can
     have the values "A","B" and "C" there will be such a field in the MockupModel.

    Deriving sub-models: works just as you'd expect.

    Aggregations: returns always the same: a vector has 0 for numerical fields and "foo" for categorical fields.

    Density: returns always 0.

    Samples: returns the vector that has the first/smallest element of each fields domain as its elements.


    It returns 0 for numerical and "foo" for categorical fields, no matter what the aggregations or queries are.
    Density queries also return constant 0.
    """

    def __init__(self, name):
        super().__init__(name)
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }

    def _fit(self, df):
        self.data = df
        self.fields = MockupModel._get_header(self.data)
        return self.update()

    def update(self):
        self._update()
        return self

    def _conditionout(self, remove):
        removeidx = self.asindex(remove)
        keepidx = utils.invert_indexes(removeidx, self._n)
        self.fields = [self.fields[i] for i in keepidx]
        return self.update()

    def _marginalizeout(self, keep):
        keepidx = self.asindex(keep)
        self.fields = [self.fields[idx] for idx in keepidx]
        return self.update()

    def _density(self, x):
        return 0

    def _maximum(self):
        return [0 if field['dtype'] == 'numerical' else 'foo' for field in self.fields]

    def _sample(self):
        value = []
        for field in self.fields:
            domain_value = field['domain'].bounded(field['extent']).value
            value.append(domain_value[0])
            # if field['dtype'] == 'numerical':
            #
            # else:

        return [0 if field['dtype'] == 'numerical' else 'foo' for field in self.fields]

    def copy(self, name=None):
        return self._defaultcopy(name)

    def _generate_model(self, opts):
        pass

