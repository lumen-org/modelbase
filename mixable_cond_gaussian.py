# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import logging

import models as md

# setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MixableCondGaussianModel(md.Model):
    """This is Conditional Gaussian (CG) model that supports true marginals and conditional.

     An advantage is, that this model stays more accurate when marginal and or conditional models are derived, compared to CG model with weak marginals (WM). See cond_gaussian_wm.py for more.

    A disadvantage is that is is much more computationally expensive to answer queries against this type of model. The reason is:
     (1) that marginals of mixable CG models are mixtures of CG models, and the query has to be answered for each component of the mixture.
     (2) the number of components grows exponentially with the number of discrete random variables (RV) marginalized.

    It seems like this should be relatively straight forward. See the notes on paper.
    In essence we never throw parameters when marginalizing discrete RV, but just mark them as 'marginalized'.

    The major questions appear to be:

      * how to efficiently store what the marginalized variables are? what about 'outer interfaces' that rely on internal storage order or something? not sure if that really is a problem
      * how to determine the maximum of such a mixture? this is anyway an interesting question...!
      * can I reuse code from cond_gaussian_wm.py? I sure can, but I would love to avoid code duplication...

    """
    pass

    def __init__(self, name):
        super().__init__(name)

        self._aggrMethods = {
            'maximum': self._maximum,
            'average': self._maximum
        }

    def _set_data(self, df, drop_silently):
        # like cond_gaussian_wm.py
        self._set_data_mixed(df, drop_silently)
        return ()

    def _fit(self):
        # like cond_gaussian_wm.py
        pass

    def _marginalizeout(self, keep, remove):
        pass

    def _conditionout(self, keep, remove):
        pass

    def _density(self, x):
        pass

    def _sample(self):
        pass

    def copy(self, name=None):
        pass


