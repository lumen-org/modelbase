import copy as cp
import logging
import numpy as np
from numpy import pi, exp, matrix, ix_, nan
from sklearn import mixture

import utils
import models as md
from models import AggregationTuple, SplitTuple, ConditionTuple
import domains as dm

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ConditionallyGaussianModel(md.Model):
    """A conditional gaussian model and methods to derive submodels from it
    or query density and other aggregations of it.

    Note that all conditional Gaussians use the same covariance matrix. Their mean, however, is not identical.
    """

    def __init__(self, name):
        super().__init__(name)

        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }

    @staticmethod
    def _get_header(df):
        """ Returns suitable fields for this model from a given pandas dataframe.
        """
        raise NotImplementedError()
        #fields = []
        #for column in df:
        #    field = md.Field(column, dm.NumericDomain(), dm.NumericDomain(df[column].min(), df[column].max()), 'numerical')
        #    fields.append(field)
        # return fields

    def fit(self, df):
        """Fits the model to passed DataFrame

        Parameters:
            df: A pandas data frame that holds the data to fit the model to. All columns in the data frame will be used.
                The data frame must have its columns ordered such that categorical columns occur before continuous
                columns.

        Internal:
            This method estimates the set of mean parameters that fit best to the data given in the dataframe df.

        Returns:
            The modified model with selected parameters set.
        """
        self.data = df
        self.fields = ConditionallyGaussianModel._get_header(self.data)
        # TODO
        raise NotImplementedError()
        return self.update()

    def update(self):
        """Updates dependent parameters / precalculated values of the model after some internal changes."""
        self._update()
        raise NotImplementedError()
        return self

    def _conditionout(self, remove):
        """Conditions the random variables with name in remove on their available, //not unbounded// domain and marginalizes
                them out.

                Note that we don't know yet how to condition on a non-singular domain (i.e. condition on interval or sets).
                As a work around we therefore:
                  * for continuous domains: condition on (high-low)/2
                  * for discrete domains: condition on the first element in the domain
         """

        if len(remove) == 0 or self._isempty():
            return self
        if len(remove) == self._n:
            return self._setempty()

        # collect singular values to condition out on
        j = sorted(self.asindex(remove))
        condvalues = []
        for idx in j:
            field = self.fields[idx]
            domain = field['domain']
            dvalue = domain.value()
            assert (domain.isbounded())
            if field['dtype'] == 'numerical':
                condvalues.append(dvalue if domain.issingular() else (dvalue[1] - dvalue[0]) / 2)
                # TODO: we don't know yet how to condition on a not singular, but not unrestricted domain.
            else:
                raise ValueError('invalid dtype of field: ' + str(field['dtype']))
        condvalues = matrix(condvalues).T

        # calculate updated mu and sigma for conditional distribution, according to GM script
        i = utils.invert_indexes(j, self._n)

        raise NotImplementedError()

        self.fields = [self.fields[idx] for idx in i]
        return self.update()

    def _marginalizeout(self, keep):
        if len(keep) == self._n or self._isempty():
            return self
        if len(keep) == 0:
            return self._setempty()

        # i.e.: just select the part of mu and sigma that remains
        #keepidx = sorted(self.asindex(keep))
        # TODO
        raise NotImplementedError()
        #self.fields = [self.fields[idx] for idx in keepidx]
        return self.update()

    def _density(self, x):
        """Returns the density of the model at point x.

        Args:
            x: a list of values as input for the density.
        """
        raise NotImplementedError()

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        raise NotImplementedError()

    def _sample(self):
        raise NotImplementedError()

    def copy(self, name=None):
        name = self.name if name is None else name
        mycopy = ConditionallyGaussianModel(name)
        # todo: implement rest
        #raise NotImplementedError()
        #mycopy.update()
        #return mycopy


if __name__ == '__main__':
    import pdb
    # todo: some testing