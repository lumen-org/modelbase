# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
import copy as cp
import numpy as np
from numpy import nan
import xarray as xr
import logging

import utils
import models as md
import domains as dm

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

    @staticmethod
    def _get_header(df):
        fields = []
        for column_name in df:
            column = df[column_name]
            domain = dm.DiscreteDomain()
            extent = dm.DiscreteDomain(sorted(column.unique()))
            fields.append(md.Field(column_name, domain, extent, 'string'))
        return fields

    @staticmethod
    def _maximum_aposteriori(df, fields, k=1):
        extents = [f['extent'].value() for f in fields]
        sizes = tuple(len(e) for e in extents)
        z = np.zeros(shape=sizes)

        # initialize count array
        counts = xr.DataArray(data=z, coords=extents, dims=df.columns)

        # iterate over table and sum up occurrences
        # TODO: this is super slow... use group by over the data frame instead
        for row in df.itertuples():
            values = row[1:]  # 1st element is index, which we don't need
            counts.loc[values] += 1

        # smooth, normalize and return
        return (counts + k) / (counts.sum() + k * counts.size)

    def fit(self, df):
        self.data = df
        self.fields = CategoricalModel._get_header(self.data)
        self._p = CategoricalModel._maximum_aposteriori(df, self.fields)
        return self.update()

    # def __str__(self):
    #     return ("Multivariate Categorical Model '" + self.name + "':\n" +
    #             "dimension: " + str(self._n) + "\n" +
    #             "names: " + str([self.names]) + "\n" +
    #             "fields: " + str([str(field['name']) + ':' + str(field['domain']) + ':' + str(field['extent'])
    #                               for field in self.fields]))

    def update(self):
        """updates dependent parameters / precalculated values of the model"""
        self._update()
        if self._n == 0:
            self._p = xr.DataArray([])
        return self

    def _conditionout(self, remove):
        # Conditioning out categorical variables works by means of the definition of conditional probability:
        #   p(x|c) = p(x,c) / p(c)
        # where p(x,c) is the join probability. Hence it works by normalizing a subrange of the probability
        # look-up

        # collect singular values to condition out on
        removeidx = sorted(self.asindex(remove))
        # todo: not needed anymore: remove = [self.names[idx] for idx in j]  # reorder to match index order
        pairs = []
        for idx in removeidx:
            field = self.fields[idx]
            domain = field['domain']
            dvalue = domain.value()
            assert (domain.isbounded())
            if field['dtype'] == 'string':
                # TODO: we don't know yet how to condition on a not singular, but not unrestricted domain.
                pairs.append((field['name'], dvalue if domain.issingular() else dvalue[0]))
            else:
                raise ValueError('invalid dtype of field: ' + str(field['dtype']))

        # 1. trim the probability look-up table to the appropriate subrange
        p = self._p.loc[dict(pairs)]

        # 2. normalize
        self._p = p / p.sum()

        # 3. keep all fields not in remove
        keepidx = utils.invert_indexes(removeidx, self._n)
        self.fields = [self.fields[idx] for idx in keepidx]

        return self.update()

    def _marginalizeout(self, keep):
        keepidx = sorted(self.asindex(keep))
        removeidx = utils.invert_indexes(keepidx, self._n)
        # the marginal probability is the sum along the variable(s) to marginalize out
        self._p = self._p.sum(dim=[self.names[idx] for idx in removeidx])
        self.fields = [self.fields[idx] for idx in keepidx]
        return self.update()

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
        mycopy.update()
        return mycopy

if __name__ == '__main__':
    import pdb
    import pandas as pd
    df = pd.read_csv('data/categorical_dummy.csv')
    model = CategoricalModel('model1')
    model.fit(df)

    print('model:', model)
    print('model._p:', model._p)

    res = model.predict(
        predict=['Student', 'City', 'Sex',
               md.AggregationTuple(['City', 'Sex', 'Student'], 'density', 'density', [])],
        splitby=[md.SplitTuple('City', 'elements', []), md.SplitTuple('Sex', 'elements', []),
               md.SplitTuple('Student', 'elements', [])])
    print('probability table: \n', str(res))

    res = model.predict(
        predict=['City', 'Sex',
                 md.AggregationTuple(['City', 'Sex'], 'density', 'density', [])],
        splitby=[md.SplitTuple('City', 'elements', []), md.SplitTuple('Sex', 'elements', [])])
    print("\n\npredict marginal table: city, sex\n" + str(res))

    res = model.predict(
        predict=['City',
                 md.AggregationTuple(['Sex'], 'maximum', 'Sex', [])],
        splitby=[md.SplitTuple('City', 'elements', [])])
    print("\n\npredict most likely sex by city: \n" + str(res))

    res = model.predict(
        predict=['Sex',
                 md.AggregationTuple(['Sex'], 'density', 'density', [])],
        splitby=[md.SplitTuple('Sex', 'elements', [])])
    print("\n\npredict marginal table: sex\n" + str(res))

    res = model.predict(
        predict=['City', 'Sex',
                 md.AggregationTuple(['City', 'Sex'], 'density', 'density', [])],
        splitby=[md.SplitTuple('City', 'elements', []), md.SplitTuple('Sex', 'elements', [])],
        where=[md.ConditionTuple('Student', '==', 'yes')])
    print("\n\nconditional prop table: p(city, sex| student=T):\n" + str(res))

    print('\n\nafter this comes less organized output:')
    print('model density 1:', model._density(['Jena', 'M', 'no']))
    print('model density 1:', model._density(['Erfurt', 'F', 'yes']))

    print('model maximum:', model._maximum())

    marginalAB = model.copy().marginalize(keep=['City', 'Sex'])
    print('marginal on City, Sex:', marginalAB)
    print('marginal p: ', marginalAB._p)

    conditionalB = marginalAB.condition([('City', '==', 'Jena')]).marginalize(keep=['Sex'])
    print('conditional Sex|City = Jena: ', conditionalB)
    print('conditional Sex|City = Jena: ', conditionalB._p)

    print('most probable city: ', model.copy().marginalize(keep=['City']).aggregate('maximum'))
    print('most probably gender in Jena: ',
          model.copy().condition([('City', '==', 'Jena')]).marginalize(keep=['Sex']).aggregate('maximum'))
    print('most probably Sex for Students in Erfurt: ',
          model.copy().condition([('City', '==', 'Erfurt'), ('Student', '==', 'yes')]).marginalize(keep=['Sex']).aggregate('maximum'))
