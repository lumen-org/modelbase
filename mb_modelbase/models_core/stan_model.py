#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13:49:15 2019

@author: Christian Lengert
@email: christian.lengert@dlr.de
"""

import pystan
import dill as ser
import pandas as pd
import multiprocessing as mp
from collections.abc import Iterable

# import pickle as ser

import copy as cp
from mb_modelbase.models_core.models import Model


# Distinguish
# Observable/Unobservable => predictive
# Independance
#


def compile_and_pickle(stan_file, output_file):
    sm = pystan.StanModel(stan_file)
    with open(output_file, "wb") as f:
        ser.dump(sm, f)


class StanPPLModel(Model):
    def __init__(
        self,
        name,
        stan_file=None,
        model_pickle_file=None,
        stan_model=None,
        evidence=None,
        n_chains=4,
        control={
            "adapt_engaged": True,  # Default True
            "adapt_gamma": 0.05,  # Default 0.05
            "adapt_delta": 0.8,  # Default 0.8
            "adapt_kappa": 0.75,  # Default 0.75
            "adapt_t0": 10,  # Default 10
        },
    ):
        super().__init__(name)
        self.control = control
        self.evidence = evidence

        if n_chains is None:
            self.n_chains = mp.cpu_count()
        else:
            self.n_chains = n_chains

        # Load pickled mode xor compile stan_file
        #        if stan_file is None and model_pickle_file is None and stan_model is None:
        #            raise ValueError("Provide either stan_file model_pickle_file, not both.")

        if stan_model is not None:
            self.stan_model = stan_model
        elif model_pickle_file is not None:
            with open(model_pickle_file, "rb") as f:
                f.seek(0)
                self.stan_model = ser.load(f)
        elif stan_file is not None:
            self.stan_model = pystan.StanModel(stan_file)

        self._marginalized = {}
        self._conditioned = {}

        # extract model structure from stan_model by looking at a trace
        samples = self._sample_stan(data=evidence)
        df = self._samples_to_df(samples)
        self.set_data(df, drop_silently=False)

    def _sample_stan(self, data):
        return self.stan_model.sampling(
            data=data, chains=self.n_chains, control=self.control, verbose=False
        ).extract()

    def _samples_to_df(self, samples):

        """
        This function takes a STAN sampling trace and builds a dataframe suitable for the modelbase
        """

        def has_multiple_instances(stan_variable_data):
            """An STAN variable has only a single instance if the"""
            return isinstance(stan_variable_data[0], Iterable)

        multiple_instances = set(
            [i for i in samples.keys() if has_multiple_instances(samples[i])]
        )
        single_instances = set(samples.keys()) - multiple_instances

        # Add variables with only a single instance
        result_dict = {k: samples[k] for k in single_instances}

        # Add variables with multiple instances where each instance becomes a column
        for var_name in multiple_instances:
            result_dict.update(
                {
                    var_name + "_" + str(i): samples[var_name][:, i]
                    for i in range(samples[var_name].shape[1])
                }
            )

        df = pd.DataFrame(result_dict)
        return df

    def __str__(self):
        return "{}\nkde_bandwidth={}".format(super().__str__(), self.kde_bandwidth())

    def _set_data(self, df, drop_silently, **kwargs):
        self._set_data_mixed(df, drop_silently)
        """"""

    def _fit(self):
        """"""

    def _marginalizeout(self, keep, remove):
        self._marginalized.update(remove)
        """"""

    def _conditionout(self, keep, remove):
        """"""

    def _density(self, x):
        """"""

    def _negdensity(self, x):
        return -1 * self._density(x)

    def _sample(self, n, mode="original", pp=False, sample_mode="first"):
        return (
            self._samples_to_df(self._sample_stan(self.evidence))
            .iloc[0:n, :]
            .drop(self._marginalized, axis=1)
        )

        """"""

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy.stan_model = cp.deepcopy(self.stan_model)
        return mycopy

    def _maximum(self):
        """"""


if __name__ == "__main__":
    import pickle

    # Compile model and pickle to file
    stan_file = "/home/me/git/lumen/datasets/mb_data/eight_schools/eight_schools.stan"
    output_file = (
        "/home/me/git/lumen/datasets/mb_data/eight_schools/eight_schools.stan.pickle"
    )

    # spm = StanPPLModel(name="stan_model_eight_schools", stan_file=stan_file)
    # with open(output_file, "wb") as f:
    #    pickle.dump(spm, f)

    compile_and_pickle(stan_file, output_file)
