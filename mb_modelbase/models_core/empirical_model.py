# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr


class EmpiricalModel(Model):
    """
    An empirical model is a model that is directly based on the relative frequency of relevant evidence.
    """

    def __init__(self, name):
        super().__init__(name)
        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }
        self._emp_data = None

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self, **kwargs):
        """Fits the model to data of the model
        Returns:
            Empty tuple, since not callback is required.
        """
        # fitting simply consists of setting the data...
        self._emp_data = self.data
        return ()

    def __str__(self):
        return ("Emperical Model '" + self.name + "':\n" +
                "dimension: " + str(self.dim) + "\n" +
                "names: " + str([self.names]) + "\n" +
                "fields: " + str([str(field) for field in self.fields]))

    def _condition(self, conditions):
        """Conditions it according to conditions"""
        self._emp_data = data_op.condition_data(self._emp_data, conditions)
        return ()

    def _conditionout(self, keep, remove):
        """Conditions the random variables with name in remove on their available, //not unbounded// domain and marginalizes them out.
         """
        # TODO: I believe this method will not be required for this model class at all.
        # because it will not be part of the required abstract interface of a model class anymore

        # collect conditions
        values = [self.byname(r)['domain'].values() for r in remove]
        conditions = zip(remove, ['in']*len(values), values)

        # condition
        self._condition(conditions)

        # marginalize out
        self._marginalizeout(keep, remove)
        return ()

    def _marginalizeout(self, keep, remove):
        """Marginalizes the dimensions in remove, keeping all those in keep"""
        self._emp_data = self._emp_data.loc[:, keep]
        return ()

    def _density(self, x):
        """Returns the density at x"""
        return data_op.density(self._emp_data, x)

    def _probability(self, event):
        """Returns the probability of event"""
        return data_op.probability(self._emp_data, event)

    def _maximum(self):
        """Returns the point of the maximum density"""
        return data_aggr.aggregate_data(self._emp_data, 'maximum')

    def _sample(self, n):
        """Returns random point of evidence"""
        return self._emp_data.sample(n, replace=True)

    def copy(self, name=None):
        """Returns a copy of this model."""
        mycopy = self._defaultcopy(name)
        mycopy._emp_data = self._emp_data
        return mycopy

    def _generate_model(self, opts):
        raise NotImplementedError
