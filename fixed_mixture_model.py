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

# list of attributes names of Model class to update
_attrs_to_update = ('fields', 'names', 'data', 'dim', '_name2idx', 'mode', '_modeldata_field')


class FixedMixtureModel(md.Model):
    """An abstract mixture model, i.e. a compound model that consists of n model types (models[i].class) with a
    particular model class occuring models[i].n many times.

    # Reduction to compontent models:
    It is however easy to implement since it is straighforward to reduce most of mixture models functionality to the
     (provided) functionality of its individual components:

     conditioning, marginalizing, density, copy

    Fitting a mixture model and querying its maximum, however, is not generically reductable. Hence this is must be
    implemented by specific mixture subclasses.

    # Performance note:
    This is a naive implementation of mixture models with relatively large overhead since each of the models maintains
     its own set of state variables for abstract model variables like model.data or model.fields.

    # Ideas to improve performance:
    The largest performance overhead probably stems from duplicated data and fields.

    data: We can let all models use the same dataframe. And we can maybe(?) use the 'mode' variable to disable
     data computations on all but one (e.g. the first) component model.

    fields: Somehow we need to let all models use the same set of fields, and only update them once...

    However, notice that each submodel nevertheless needs to maintain its own set of auxiliary variables which
    are updated by calls to update(). update() in turn calls _update() which updates e.g. a models fields mappings
    and such.

    I think it should be possible to manually override the mapping of fields, data and such after the creation
    of the mixture model. Also, I can override the function binding of update for all but one model and hence assure
    that all components use the same variables, it is only updated once.

    Problem might be: I have to condition out, marginalize out on all components, and then after finishing with that
    update the models fields. Lets just start and see how it goes.

    # Notes:
    Since this class inherits from Model it actually maintains its own set of field and data. We should simply map
    all models data and fields to this one!

    Problem: we can link many, but not all, since not all are obj. Actually we can link to the objects, but usually
      the objects itself don't change but the reference is assigned a new object. Therefore not even in the case of
      objects this is easy to handle.

    Solution Idea: Assign getter and setter to each of components attributes. See http://stackabuse.com/python-properties/

    Solution Idea2: ah, just don't care about it, at least for fields etc. Only avoid copying data many times!

    Avoid premature optimization!


    ## Instance Attributes

    components:
        meaning: The mixtures models components.
        data structure: a dict that maps the class name (MyModelClass.__name__) to a tuple of actual models of that
            class
    """
    @staticmethod
    def create_components(models):
        """Create requested number of models of each model class and store them in a dict that maps class name to
         tuple of dicts.
         """
        components = dict()
        for class_, count in models.items():
            class_name = class_.__name__
            instances = tuple(class_(str(class_name) + str(i)) for i in range(count))
            components[class_name] = instances
        return components

    def __init__(self, name, models):
        """Constructs a new instance of a fixed mixture model.

        Args:
            name: name of the model

        Note: You generally need to call set_models on that instance to fill it with actual components!
         """
        # TODO: combine set_models and __init__ into one. Problem: _defaultcopy
        super().__init__(name)
        self._models = None
        self.components = None

    def __iter__(self):
        """Returns an iterator over all component models of this mixture model."""
        for models_per_class in self.components:
            for model in models_per_class:
                yield model

    def set_models(self, models):
        """ Args:
            models: a sequence of pairs (model_class, k), where model_class is the model class of the components
                and k is the number of model instances to be used for that.
        """
        self._models = models  # just to store it
        self.components = self.create_components(models)
        self._update_in_components()

    def _update_in_components(self, attrs_to_update = _attrs_to_update):
        """Update dependent variables/states in all components to avoid recalculation / duplicated storage.
        """
        for model in self:
            for attr in attrs_to_update:
                setattr(model, attr, getattr(self, attr))

    def __str__(self):
        # todo: add information about mixture components
        return ("Fixed Mixture Model '" + self.name + "':\n" +
                "dimension: " + str(self.dim) + "\n" +
                "names: " + str([self.names]) + "\n" +
                "fields: " + str([str(field) for field in self.fields]) + "\n")
        #+
        #        "\n".join([]))

    def update(self):
        """Updates dependent parameters / precalculated values of the model"""
        #self._update()
        # TODO: should this method call _update_in_components ?
        return self

    def _conditionout(self, keep, remove):
        for model in self:
            pass
        return self.update

    def _marginalizeout(self, keep, remove):
        #return self.update()
        return self.update

    def _density(self, x):
        return 0

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)

        # copy all models
        for class_name, models_per_class in self.components.items():
            mycopy.components[class_name] = tuple(model.copy() for model in models_per_class)

        # update/link all
        mycopy.update()
        mycopy._update_in_components()
        return mycopy

    def _set_data(self, df, drop_silently):
        # cannot generically implement it, because I don't know which _set_data I should call
        # however, don't forget to update after setting the data
        raise NotImplementedError("Implement this method in your subclass")

    def _maximum(self):
        raise NotImplementedError("Implement this method in your subclass")

    def _fit(self):
        raise NotImplementedError("Implement this method in your subclass")

    def _sample(self):
        raise NotImplementedError("Implement this method in your subclass")

    def _generate_model(self, opts):
        raise NotImplementedError("Implement this method in your subclass")
