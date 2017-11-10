# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)

import logging
import functools
import math

from ..utils import utils
from . import models as md
from . import domains as dm

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MockUpModel(md.Model):
    """A mock up model solely for testing of the abstract Model class.

    You can use it as any other model class, i.e. you can fit data to it, marginalize, aggregate, density-query, ...
    it.

    Fitting: No actual fitting will be done, however, suitable fields for the model are derived from the data. The
     Fields actually match whatever you provide as data, e.g. if you provide data with a categorical column that can
     have the values "A","B" and "C" there will be such a field in the MockupModel.

    Deriving sub-models: works just as you'd expect.

    Aggregations: returns the vector that has the first/smallest element of each fields domain as its elements.

    Density: returns always 0.

    Samples: returns the vector that has the first/smallest element of each fields domain as its elements.
    """

    def __init__(self, name="no-name"):
        super().__init__(name)
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        self._unbound_updater = functools.partial(self.__class__._update, self)

    def _set_data(self, df, drop_silently):
        self._set_data_mixed(df, drop_silently)
        return self._unbound_updater,

    def _fit(self):
        return self._unbound_updater,

    def _update(self):
        return self

    def _conditionout(self, keep, remove):
        return self._unbound_updater,

    def _marginalizeout(self, keep, remove):
        return self._unbound_updater,

    def _density(self, x):
        return 0

    def _maximum(self):
        return self._sample()

    def _sample(self):
        values = []
        for field in self.fields:
            domain = field['domain']
            val = domain.bounded(field['extent']).value()
            values.append(val if domain.issingular() else val[0])
        return values

    def copy(self, name=None):
        return self._defaultcopy(name)

    def _generate_model(self, opts={}):
        """opts has a optional key 'dim' that defaults to 6 and specifies the dimension of the model that you want to
        have."""
        if 'dim' not in opts:
            dim = 6
        else:
            dim = opts['dim']

        ncat = math.floor(dim/2)
        nnum = dim - ncat
        self.fields = []

        # numeric
        for idx in range(ncat):
            field = md.Field(name="dim" + str(idx),
                             domain=dm.NumericDomain(),
                             extent=dm.NumericDomain(-5, 5))
            self.fields.append(field)

        # categorical
        for idx in range(ncat):
            field = md.Field(name="dim" + str(idx+nnum),
                             domain=dm.DiscreteDomain(),
                             extent=dm.DiscreteDomain(list("ABCDEFG")))
            self.fields.append(field)

        self.mode = 'model'
        return self._unbound_updater,

if __name__ == "__main__":
    model = MockUpModel()
    model.generate_model(opts={'dim': 4})
    pass
