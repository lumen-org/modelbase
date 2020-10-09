# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import functools
import numpy as np
from numpy import nan
import xarray as xr
import logging

from ..utils import utils
from . import models as md

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class CategoricalModel(md.Model):
    """A multivariate categorical model"""
    def __init__(self, name):
        super().__init__(name)
        self._p = nan
        self._aggrMethods = {
            'maximum': self._maximum
        }
        # creates an self contained update function. we use it as a callback function later
        self._unbound_updater = functools.partial(self.__class__._update, self)

    @staticmethod
    def _maximum_aposteriori(df, fields, k=1):
        extents = [f['extent'].value() for f in fields]
        sizes = tuple(len(e) for e in extents)
        z = np.zeros(shape=sizes)

        # initialize count array
        counts = xr.DataArray(data=z, coords=extents, dims=df.columns)

        # iterate over table and sum up occurrences
        for row in df.itertuples():
            values = row[1:]  # 1st element is index, which we don't need
            counts.loc[values] += 1

        # smooth, normalize and return
        return (counts + k) / (counts.sum() + k * counts.size)

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_categorical(df, drop_silently)
        return ()

    def _fit(self):
        self._p = CategoricalModel._maximum_aposteriori(self.data, self.fields)
        return self._unbound_updater,

    def _update(self):
        """updates dependent parameters / precalculated values of the model"""
        if self.dim == 0:
            self._p = xr.DataArray([])
        return self

    def _conditionout(self, keep, remove):
        # Conditioning out categorical variables works by means of the definition of conditional probability:
        #   p(x|c) = p(x,c) / p(c)
        # where p(x,c) is the join probability. Hence it works by normalizing a subrange of the probability
        # look-up

        # collect singular values to condition out on
        pairs = dict(self._condition_values(remove, True))

        # 1. trim the probability look-up table to the appropriate subrange
        p = self._p.loc[pairs]

        # 2. normalize
        self._p = p / p.sum()

        # 3. keep all fields not in remove
        return self._unbound_updater,

    def _marginalizeout(self, keep, remove):
        keepidx = sorted(self.asindex(keep))
        removeidx = utils.invert_indexes(keepidx, self.dim)
        # the marginal probability is the sum along the variable(s) to marginalize out
        self._p = self._p.sum(dim=[self.names[idx] for idx in removeidx])  # TODO: cant I use integer indexing?!
        return self._unbound_updater,

    def _density(self, x):
        # note1: need to convert x to tuple for indexing
        # note2: .values.item() is to extract the scalar as a float
        return self._p.loc[tuple(x)].values.item()

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        # todo: how to directly get the coordinates of the maximum?
        p = self._p
        pmax = p.where(p == p.max(), drop=True)  # get view on maximum (coordinates remain)
        return [idx[0] for idx in pmax.indexes.values()]  # extract coordinates from indexes

    def _sample(self):
        raise NotImplementedError()

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy._p = self._p
        mycopy._update()
        return mycopy

if __name__ == '__main__':
    pass
