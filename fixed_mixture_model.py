# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import functools
import logging
import math

import models as md

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# list of attributes names of Model class to update
_attrs_to_update = ('fields', 'names', 'data', 'dim', '_name2idx', 'mode', '_modeldata_field')


class FixedMixtureModel(md.Model):
    """An abstract mixture model, i.e. a compound model that consists of n model types (models[i].class) with a
    particular model class occurring models[i].n many times.

    # Reduction to compontent models:
    It is however easy to implement since it is straightforward to reduce most of mixture models functionality to the
     (provided) functionality of its individual components:

      * conditioning
      * marginalizing
      * density
      * copy

    Other functionality, however, is not generically reducible. Hence this is must be implemented by specific mixture subclasses:

      * fitting a mixture model
      * setting its data
      * aggregations

    However, for the maximum aggregation there is a simple, generic heuristic implemented. See _maximum_naive_heuristic.

    # internal performance note:
    This is a naive implementation of mixture models that tries to reuse as much of existing model classes as possible.
     It avoids duplicated storage of a models data and other state variables of an abstract model. This works by
     linking the the components state to the state of the fixed_mixture_model, which inherits from model and hence
     maintains it own state.
     It also uses the internal model api, i.e. _marginalizeout and _conditionout of the components, as we can make
      guarantees on the correctness of the parameters.

    However, notice that each component nevertheless needs to maintain its own set of internal auxiliary variables.

    Future: A discussion with Matthias led to the conclusion that we probably should create independent classes
    for data and fields. This would improve code quality and would most likely also help to manage these in the
    context of mixture models.

    # Instance Attributes

    components:
        meaning: The mixtures models components.
        data structure: a dict that maps the class name (MyModelClass.__name__) to a tuple of actual models of that
            class

    _unbound_component_updater:
        function that can be called to maintain internal state. Call to it is equivalent to
        calling self._update_in_components()

    [i]:
        meaning: access individual mixture components by numerical indexing.

    len():
        meaning: get the number of components by calling len on the mixture.

    iter:
        meaning: iterate over the mixture to access the components one after another.
    """
    @staticmethod
    def create_components(models):
        """Create requested number of models of each model class and store them in a dict that maps class name to
         tuple of dicts.
         """
        components = dict()
        for class_, count in models:
            class_name = class_.__name__
            instances = tuple(class_(str(class_name) + str(i)) for i in range(count))
            components[class_name] = instances
        return components

    def __init__(self, name):
        """Constructs a new instance of a fixed mixture model.

        Note: You generally need to call set_models on that instance to fill it with actual components!

        Args:
            name: name of the model
         """
        # TODO: combine set_models and __init__ into one. Problem: _defaultcopy
        super().__init__(name)
        self.components = {}
        # creates an self contained update function. we use it as a callback function later
        self._unbound_component_updater = functools.partial(self.__class__._update_in_components, self)

    def __iter__(self):
        """Returns an iterator over all component models of this mixture model."""
        for models_per_class in self.components.values():
            for model in models_per_class:
                yield model

    def __len__(self):
        """Returns the number of components of the mixture model."""
        return len(self._component_sequence)

    def __getitem__(self, idx):
        """Supports numeric indexing to access component models."""
        return self._component_sequence[idx]

    def _set_models(self, models):
        """ Fills the mixture with (empty) models.
         Args:
            models: a sequence of pairs (model_class, k), where model_class is the model class of the components
                and k is the number of model instances to be used for that.
        """
        self.components = self.create_components(models)
        self._component_sequence = [model for model in self]
        self.weights = [None] * len(self)
        self._update_in_components()

    def _update_in_components(self, attrs_to_update = _attrs_to_update):
        """Update dependent variables/states in all components to avoid recalculation / duplicated storage.
        By default it updates all relevant attributes of an abstract model. This, however, can be customized.
        """
        for model in self:
            for attr in attrs_to_update:
                setattr(model, attr, getattr(self, attr))

    def _conditionout(self, keep, remove):
        callbacks = [self._unbound_component_updater]
        for conditions in (model._conditionout(keep, remove) for model in self):
            callbacks.extend(conditions)
        return callbacks

    def _marginalizeout(self, keep, remove):
        callbacks = [self._unbound_component_updater]
        for conditions in (model._marginalizeout(keep, remove) for model in self):
            callbacks.extend(conditions)
        return callbacks

    def _density(self, x):
        return sum(model._density(x) * weight for weight, model in zip(self.weights, self))

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy.weights = self.weights[:]

        # copy all models
        for class_name, models_per_class in self.components.items():
            mycopy.components[class_name] = tuple(model.copy() for model in models_per_class)

        # update/link all
        mycopy._update_in_components()
        return mycopy

    def _sample(self):
        # choose component by weight

        # sample from it

        # TODO: can be done generically!
        raise NotImplementedError("Implement it - this can be done generically!")

    def _set_data(self, df, drop_silently):
        # we need to link the set data to all components, hence we need to run _update_in_components after setting
        # data in all mixture models
        # the actual data setting, however, cannot be generically implemented
        callbacks = self._set_data_4mixture(df, drop_silently)
        return (self._unbound_component_updater,) + callbacks

    def _maximum_naive_heuristic(self):
        # this is an pretty stupid heuristic :-)
        maximum = None
        maximum_density = -math.inf

        for weight, model in zip(self.weights, self):
            cur_maximum = model._maximum()
            cur_density = model._density(cur_maximum)*weight
            if cur_density > maximum_density:
                maximum = cur_maximum
                maximum_density = cur_density

        return maximum

    def _set_data_4mixture(self):
        raise NotImplementedError("Implement this method in your subclass")

    def _maximum(self):
        # cannot be done generically
        raise NotImplementedError("Implement this method in your subclass")

    def _fit(self):
        # cannot be done generically
        raise NotImplementedError("Implement this method in your subclass")

    def _generate_model(self, opts):
        # cannot be done generically
        raise NotImplementedError("Implement this method in your subclass")
