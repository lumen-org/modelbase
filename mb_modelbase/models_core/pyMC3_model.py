# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)
# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
# Copyright (c) 2019 Jonas Gütter (jonas.aaron.guetter@uni-jena.de)

import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as sciopt
import copy as cp
from functools import reduce
import theano
import math

from mb_modelbase.models_core.models import Model
from mb_modelbase import utils
from mb_modelbase.utils.data_import_utils import get_numerical_fields
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.utils.data_type_mapper import DataTypeMapper

from scripts.julien.sampler.graphical_model_sampling import gen_samples_for_model


class ProbabilisticPymc3Model(Model):
    """A Bayesian model built using the PyMC3 library.

    General principles:

    The heart of this class is the generation of posterior samples during the _sample() method. It uses the PyMC3
    functions sample() and sample_ppc() to generate samples for both latent and observed random variables, and store
    them in a pandas dataframe where columns are variables and rows are associated samples. Fitting the
    model essentially means just generating those samples. Computing a probability density for the model is then done by
    setting up a kernel density estimator over the samples with the scipy.stats library and getting the density from
    this estimator. Marginalizing means just dropping columns from the dataframe, conditioning means just dropping
    rows that do not meet a certain condition. Three things to keep in mind for the conditioning:
        - Conditioning in the actual sense of conditioning on one single value is not feasible here: If we have
          continuous variables, none of the samples probably has this value, so all samples would be discarded. However
          for the current implementation it is only necessary to condition on intervals, so that is currently not a
          problem
        - However a similar problem appears if we have a high-dimensional model and want to condition on a small area
          within that high-dimensional space. Then it is very likely that again none of the samples fall in this space
          and again, all samples would be discarded. We do not have a solution for this yet.
        - Keep in mind that the _conditionout() method does not only condition, but additionally marginalizes variables
          out. There is no method that only conditions the model

    A model is allowed to include independent variables, that is, variables that are not modeled but whose values
    influence observed variables. If a model contains independent variables, values for them are generated after
    the latent variables were sampled and before the observed variables are sampled. The values for the independent
    variables are drawn without replacement from the training data. Setting new values for independent variables is not
    so easy, since they are actually hardcoded within the PyMC3 model, which cannot easily be changed once submitted
    to the Wrapper class of the modelbase backend. To work around this, independent variables have to be transformed to
    theano shared variables and given to the wrapper class as a separate argument. Then it is possible to change their
    values again after the model was specified.

    Parameters:

        model_structure : a PyMC3 Model() instance

        shared_vars : dictionary of theano shared variables

            If the model has independent variables, they have to be encoded as theano shared variables and provided
            in this dictionary, additional to the general dataframe containing all observed data. Watch out: It is
            NOT guaranteed that shared_vars always holds the original independent variables, since they are changed
            during the _fit()-method

        nr_of_posterior_samples: integer scalar specifying the number of posterior samples to be generated

        fixed_data_length: boolean, indicates if the model requires the data to have a fixed length

            Some probabilistic models require a fixed length of the data. This is important because normally
            new data points are generated with a different length than the original data

        data_mapping: dict or DataTypeMapper, optional. Defaults to the identify mapping.

            The actual probabilistic modelling may require that you encode variables differently than you want to
            expose them to the outside. E.g. a variable may be modelled as an integer value of 0 or 1 but actually
            it represents the sex of a person ('male' or 'female'). Here you specify such mappings between the
            original space and the modeling space.

            Note that you must provide your training and test data in its original space representation with
            .fit_data().

        sampling_chains: int, optional. Defaults to 1.

            See https://docs.pymc.io/api/inference.html and there the parameter chains of .sample()

        sampling_cores: int, optional. Defaults to 1.

            See https://docs.pymc.io/api/inference.html and there the parameter cores of .sample()

        probabilistic_program_graph: dict, optional. Defaults to None.

            The graph of the probabilistic program this model is based on. This only makes sense in the special
            case where the PyMC3 program is derived automatically from a bayesian network. The graph represents
            the data flow in the network, but also contains information about the user defined constrains that
            were taken into account when learning the bayesian network. It is a dict organized as follows:

            'nodes': list of strings.
                Name of every node of the graph. It includes all nodes, also the enforced ones below.

            'edges': 2-tuple of strings
                Directed edges of the graph as pairs of nodes. It includes all edges, even the forbidden ones.

            'enforced_node_dtypes': dict of <node name: dtype>, where dtype maybe 'numerical' or 'string'. Optional.

            'enforced_edges': list of edges. Optional.

            'forbidden_edges': list of edges. Optional.
        """

    def __init__(self, name, model_structure, shared_vars=None, nr_of_posterior_samples=5000,
                 fixed_data_length=False, data_mapping=None, sampling_chains=1, sampling_cores=1,
                 probabilistic_program_graph=None, sample_prior_predictive=False):
        """Bayesian models built by the PyMC3 library

        Parameters:

            model_structure : a PyMC3 Model() instance

            shared_vars : dictionary of theano shared variables

                If the model has independent variables, they have to be encoded as theano shared variables and provided
                in this dictionary, additional to the general dataframe containing all observed data. Watch out: It is
                NOT guaranteed that shared_vars always holds the original independent variables, since they are changed
                during the _fit()-method

            nr_of_posterior_samples: integer, defaults to 5000.

                The number of posterior samples to be generated, that is, this is the number of
                samples that is used to approximate the true distribution as represented by the
                underlying sampler.

            fixed_data_length: boolean, indicates if the model requires the data to have a fixed length

                Some probabilistic models require a fixed length of the data. This is important because normally
                new data points are generated with a different length than the original data

            data_mapping: dict or DataTypeMapper, optional. Defaults to the identify mapping.

                The actual probabilistic modelling may require that you encode variables differently than you want to
                expose them to the outside. E.g. a variable may be modelled as an integer value of 0 or 1 but actually
                it represents the sex of a person ('male' or 'female'). Here you specify such mappings between the
                original space and the modeling space.

                Note that you must provide your training and test data in its original space representation with
                .fit_data(). For example: your data (original space) contain a variable 'sex' with
                values 'male' and 'female' that you need internally inside your model as 0, 1.
                Hence you use a data mapper with a forward map {'male': 0, 'female': 1}. When you
                fit the model you provide the original data (that is with 'male' and 'female').

            sampling_chains: int, optional. Defaults to 1.

                See https://docs.pymc.io/api/inference.html and there the parameter chains of .sample()

            sampling_cores: int, optional. Defaults to 1.

                See https://docs.pymc.io/api/inference.html and there the parameter cores of .sample()

            probabilistic_program_graph: dict, optional. Defaults to None.

                The graph of the probabilistic program this model is based on. This only makes sense in the special
                case where the PyMC3 program is derived automatically from a bayesian network. The graph represents
                the data flow in the network, but also contains information about the user defined constrains that
                were taken into account when learning the bayesian network. It is a dict organized as follows:

                'nodes': list of strings.
                    Name of every node of the graph. It includes all nodes, also the enforced ones below.

                'edges': 2-tuple of strings
                    Directed edges of the graph as pairs of nodes. It includes all edges, even the forbidden ones.

                'enforced_node_dtypes': dict of <node name: dtype>, where dtype maybe 'numerical' or 'string'. Optional.

                'enforced_edges': list of edges. Optional.

                'forbidden_edges': list of edges. Optional.

        """
        super().__init__(name)
        self.model_structure = model_structure
        self.samples = pd.DataFrame()
        self._aggrMethods = {
            'maximum': self._maximum
        }
        self.parallel_processing = False
        self.sampling_chains = sampling_chains
        self.sampling_cores = sampling_cores
        self.shared_vars = shared_vars
        self.nr_of_posterior_samples = nr_of_posterior_samples
        self.fixed_data_length = fixed_data_length

        if data_mapping is None:
            data_mapping = {}
        if type(data_mapping) is DataTypeMapper:
            self._data_type_mapper = data_mapping
        elif type(data_mapping) is dict:
            self._data_type_mapper = DataTypeMapper()
            for name, forward_mapping in data_mapping.items():
                self._data_type_mapper.set_map(name, forward_mapping, backward='auto')

        if probabilistic_program_graph:
            utils.normalize_pp_graph(probabilistic_program_graph)
        self.probabilistic_program_graph = probabilistic_program_graph

        self._update_samples_model_representation()
        self.samples = None
        self.sample_prior_predictive = sample_prior_predictive

        self._prefetched_samples = None

    def _set_data(self, df, drop_silently, **kwargs):
        assert df.index.is_monotonic, 'The data is not sorted by index. Please sort data by index and try again'
        # Add column with index to df for later resorting
        df['index'] = df.index
        self._set_data_mixed(df, drop_silently, split_data=False)
        # Sort data by original index to make it consistent again with the shared variables
        # The other way (changing the shared vars to be consistent with the data) does not
        # work since dependent variables would not be changed then
        self.data.sort_values(by='index', inplace=True)
        self.data.set_index('index', inplace=True)
        df.set_index('index', inplace=True)
        self._update_all_field_derivatives()
        self._update_remove_fields(to_remove=['index'])
        self._update_all_field_derivatives()
        # Enforce usage of theano shared variables for independent variables
        # Independent variables are those variables which appear in the data but not in the RVs of the model structure
        model_vars = [str(name) for name in self.model_structure.basic_RVs]
        ind_vars = [varname for varname in self.data.columns.values if varname not in model_vars]
        # When there are no shared variables, there should be no independent variables. Otherwise, raise an error
        if not self.shared_vars:
            assert len(ind_vars) == 0, \
                'The model appears to include the following independent variables: ' + str(ind_vars) + ' It is '\
                'required to pass the data for these variables as theano shared variables to the ' \
                'ProbabilisticPymc3Model constructor'
        # When there are shared variables, there should be independent variables. Otherwise, raise an error
        else:
            assert len(ind_vars) > 0, ' theano shared variables were passed to the ProbabilisticPymc3Model constructor'\
                                      ' but the model does not appear to include independent variables. Only pass '\
                                      'shared variables to the constructor if the according variables are independent'
            # Each independent variable should appear in self.shared_vars. If not, raise an error
            missing_vars = [varname for varname in ind_vars if varname not in self.shared_vars.keys()]
            assert len(missing_vars) == 0, \
                'The following independent variables do not appear in shared_vars:' + str(missing_vars) + ' Make sure '\
                'that you pass the data for each independent variable as theano shared variable to the constructor'
        self.check_data_and_shared_vars_on_equality()
        # Set number of samples to the number of data points. The number of samples must not be chosen
        # arbitrarily, since independent and dependent variables have to have the same dimensions. Otherwise,
        # some models cannot compute posterior predictive samples
        # TODO: fix this
        #self.nr_of_posterior_samples = len(df)
        return ()

    def _generate_samples_for_independent_variable(self, key, size):
        lower_bound, upper_bound = self.byname(key)['extent'].value()
        generated_samples = np.linspace(lower_bound, upper_bound, num=size)
        # If the samples have another data type than the original data, problems can arise. Therefore,
        # data types of the new samples are changed to the dtypes of the original data here
        if str(generated_samples.dtype) != self.shared_vars[key].dtype:
            generated_samples = generated_samples.astype(self.shared_vars[key].dtype)
        return generated_samples

    def _update_samples_model_representation(self, recreate_samples_model_repr=True):
        if recreate_samples_model_repr:
            self._samples_model_repr = self._data_type_mapper.forward(self.samples, inplace=False)

        # needed for density calculation
        kde_input = self._samples_model_repr.values.T.astype(float)
        # require _multiple_ inputs. 5 is a heuristic to prevent singular matrices due to all identical input
        if kde_input.size > 5:
            self._samples_kde = stats.gaussian_kde(kde_input+np.eye(*kde_input.shape)+1e-9)
        else:
            self._samples_kde = None

        # needed for maximum calculation
        self._samples_model_repr_mean = self._samples_model_repr.mean(axis='index')

    def _cartesian_product_of_samples(self, df):
        # Split up df in its columns
        dfs = [pd.DataFrame(df[col]) for col in df]
        # Add equal column to each split
        [df.insert(1, 'key', 1) for df in dfs]
        # cross join all dfs
        cartesian_prod = reduce(lambda left, right: pd.merge(left, right, on='key'), dfs)
        cartesian_prod = cartesian_prod.drop('key', 1)
        return cartesian_prod

    def _fit(self):
        self.samples, self._samples_model_repr = \
            self._sample(self.nr_of_posterior_samples, mode='both',
                         pp=self.sample_prior_predictive, sample_mode='new samples')

        # Add parameters to fields
        varnames = [str(var) for var in self.model_structure.unobserved_RVs]
        for var in self.model_structure.unobserved_RVs:
            varname = str(var)
            # check if trace consists of more than one variable. Below expression is not empty when that is the case
            if var.distribution.shape:
                # Remove old varname
                varnames.remove(varname)
                # Insert new varnames
                for i in range(var.distribution.shape.item()):
                    varnames.append(varname + '_' + str(i))
            # do not add variables that already exist
            if varname in self.names:
                varnames.remove(varname)
        latent_fields = get_numerical_fields(self.samples, varnames)
        for f in latent_fields:
            f['obstype'] = 'latent'

        self.fields = self.fields + latent_fields
        self._update_all_field_derivatives()
        self._init_history()

        # Change order of sample columns so that it matches order of fields
        self.samples = self.samples[self.names]
        self._samples_model_repr = self._samples_model_repr[self.names]

        self._update_samples_model_representation(recreate_samples_model_repr=False)

        # Mark variables as independent. Independent variables are variables that appear in the data but
        # not in the observed random variables of the model
        for field in self.fields:
            if field['name'] in self.data.columns and \
                    field['name'] not in [str(var) for var in self.model_structure.basic_RVs]:
                    # field['name'] not in [str(var) for var in self.model_structure.observed_RVs]:
                field['independent'] = True
        return ()

    def _marginalizeout(self, keep, remove):
        keep_not_in_names = [name for name in keep if name not in self.names]
        if len(keep_not_in_names) > 0:
            raise ValueError('The following variables in keep do not appear in the model: ' + str(keep_not_in_names) )
        remove_not_in_names = [name for name in remove if name not in self.names]
        if len(remove_not_in_names) > 0:
            raise ValueError('The following variables in remove do not appear in the model: ' + str(remove_not_in_names))

        # Remove all variables in remove
        cols_to_remove = [col for col in remove if col in set(self.samples.columns)]
        self.samples = self.samples.drop(list(cols_to_remove), axis=1)

        self._update_samples_model_representation()
        return ()

    def _conditionout(self, keep, remove):
        keep_not_in_names = [name for name in keep if name not in self.names]
        if len(keep_not_in_names) > 0:
            raise ValueError('The following variables in keep do not appear in the model: ' + str(keep_not_in_names))
        remove_not_in_names = [name for name in remove if name not in self.names]
        if len(remove_not_in_names) > 0:
            raise ValueError('The following variables in remove do not appear in the model: ' + str(remove_not_in_names))

        values = [self.byname(r)['domain'].values() for r in remove]
        conditions = zip(remove, ['in'] * len(values), values)
        self.samples = data_op.condition_data(self.samples, conditions)
        self.samples.dropna(inplace=True)

        # is done in _marginalize_out anyway:
        # self._update_samples_model_representation()
        self._marginalizeout(keep, remove)
        return ()

    def _density(self, x):
        if any(self.fields[i]['independent'] for i in range(self.dim)):
            #raise ValueError("Density is queried for a model with independent variables")
            return np.NaN
        elif self.samples.empty | len(self.samples) == 1:
            #raise ValueError("There are not enough samples in the model")
            return np.NaN
        else:
            # map x into model space
            x = self._data_type_mapper.forward(dict(zip(self.names, x)))
            # map back to value list (in correct order)
            x = [x[name] for name in self.names]
            x = np.reshape(x, (1, -1))
            if self._samples_kde is None:
                return 0
            else:
                return self._samples_kde.evaluate(x)[0]

    def _negdensity(self, x):
        return -self._density(x)

    def _draw_new_samples(self, n, pp):
        """Draw n new samples."""
        sample = pd.DataFrame()

        # Generate samples for latent random variables
        with self.model_structure:
            if pp:
                trace = pd.DataFrame(pm.sample_prior_predictive(n))
                return self._data_type_mapper.backward(trace, inplace=False), trace
            trace = pm.sample(n, chains=self.sampling_chains, cores=self.sampling_cores,
                              progressbar=False)

        for varname in trace.varnames:
            # check if trace consists of more than one variable
            if len(trace[varname].shape) == 2:
                for i in range(trace[varname].shape[1]):
                    sample[varname + '_' + str(i)] = [var[i] for var in trace[varname]]
            else:
                sample[varname] = trace[varname]
        # Generate samples for observed independent variables
        if self.shared_vars:
            samples_independent_vars = pd.DataFrame(columns=self.shared_vars.keys())
            data_independent_vars = pd.DataFrame(columns=self.shared_vars.keys())
            for name, value in self.shared_vars.items():
                data_independent_vars[name] = value.get_value()
            # Draw without replacement from the observed data. If more values should be drawn than there are in the
            # data, take the whole data multiple times
            data_fits_in_n = math.floor(n / len(self.data))
            for i in range(data_fits_in_n):
                samples_independent_vars = samples_independent_vars.append(
                    data_independent_vars.copy())
            samples_independent_vars = samples_independent_vars.append(
                data_independent_vars.sample(n - data_fits_in_n * len(self.data), replace=False))
            # shared_vars holds independent variables, even the ones that were marginalized out earlier.
            # data holds only variables that are in the current model, but also dependent variables.
            # To get all independent variables of the current model, pick all variables that appear in both
            independent_var_names = [name for name in self.shared_vars.keys() if
                                     name in self.data.columns]
            for varname in independent_var_names:
                sample[varname] = samples_independent_vars[varname].values

        # Generate samples for observed dependent variables
        if self.shared_vars:
            shared_vars_org = {}
            for col in samples_independent_vars:
                shared_vars_org[col] = self.shared_vars[col].get_value()
                self.shared_vars[col].set_value(samples_independent_vars[col])
        with self.model_structure:
            ppc = pm.sample_ppc(trace)
            # sample_ppc works the following way: For each parameter set generated by pm.sample(), a sequence
            # of points is generated with the same length as the observed data.
            if self.shared_vars:
                for varname in self.model_structure.observed_RVs:
                    sample[str(varname)] = [ppc[str(varname)][j][j] for j in
                                            range(ppc[str(varname)].shape[0])]
            # When there are no independent variables, I cannot change the length of the sequences. So I just take
            # the first point of each sequence, which is okay since the draws are not based on any independent variables
            else:
                for varname in self.model_structure.observed_RVs:
                    sample[str(varname)] = [ppc[str(varname)][j][0] for j in
                                            range(ppc[str(varname)].shape[0])]

        # Restore independent variables to previous values. This is necessary since pm.sample()
        # requires same length of all variables and also all copies of a model use the same
        # shared variables
        if self.shared_vars:
            for col in self.shared_vars.keys():
                self.shared_vars[col].set_value(shared_vars_org[col])

        self.check_data_and_shared_vars_on_equality()
        return sample

    def _sample(self, n, mode='original', pp=False, sample_mode='first'):
        """
        Draw a sample of size n.
        :param n:
        :param mode: str. one of 'original', 'model', 'both'
            Chooses in which representation space the samples are returned.
        :param pp
            TODO document
        :param sample_mode str.
            Chooses what method to use for generating samples.
            'first' takes the first n elements of self.samples. 'choice' randomly selects n elements
            from self.samples. 'new samples' returns new drawn samples.
        :return:
        """

        # TODO: fix
        # If number of samples differs from number of data points, posterior predictive samples
        # cannot be generated
        #if n != len(self.data):
        #    print('WARNING: number of samples differs from number of data points. To avoid problems during sampling, '
        #          'number of samples is now automatically set to the number of data points')
        #    n = len(self.data)
        #n = self.nr_of_posterior_samples

        # mode 1: sample by selecting the first n of the posterior samples
        if sample_mode == 'first':
            assert(self.samples is not None)
            if n > len(self.samples):
                sample = np.tile(self.samples, np.ceil(n/len(self.samples)))[:n]
            else:
                sample = self.samples[:n]

        # mode 2: sample by choose randomly n of the posterior samples
        elif sample_mode == 'choice':
            assert (self.samples is not None)
            if n > len(self.samples):
                sample = np.random.choice(self.samples, size=n, replace=True)
            else:
                sample = np.random.choice(self.samples, size=n, replace=False)

        # mode 3: true sampling of new samples
        elif sample_mode == 'new samples':
            sample = self._draw_new_samples(n, pp)
        else:
            raise ValueError('invalid value for sample_mode: {}'.format(sample_mode))

        assert(len(sample) == n)

        # map samples from model space in to data space
        if mode is 'model':
            return sample
        elif mode is 'both':
            return self._data_type_mapper.backward(sample, inplace=False), sample
        elif mode is 'original':
            return self._data_type_mapper.backward(sample)
        else:
            raise ValueError('invalid value for mode {}'.format(mode))

    def copy(self, name=None):
        name = self.name if name is None else name
        # Note: The shared_vars attribute is not copied. Rather, the same shared_vars object is
        # used for EVERY copy of the model. Copying the object would be useless since the
        # model_structure is linked with the original object nevertheless. This means that
        # the shared vars attribute must not be changed permanently, because doing so would
        # propagate to all model copies
        # TODO: the above seems like the source of very weird future bugs that occur in race conditions ....
        mycopy = self.__class__(name, self.model_structure, self.shared_vars)
        mycopy.data = self.data.copy()
        mycopy.test_data = self.test_data.copy()
        mycopy.fields = cp.deepcopy(self.fields)
        mycopy.mode = self.mode
        mycopy._update_all_field_derivatives()
        mycopy.history = cp.deepcopy(self.history)
        mycopy.samples = self.samples.copy()
        mycopy._samples_model_repr = self._samples_model_repr.copy()
        mycopy.nr_of_posterior_samples = self.nr_of_posterior_samples
        mycopy.sampling_cores = self.sampling_cores
        mycopy.sampling_chains = self.sampling_chains
        mycopy.fixed_data_length = self.fixed_data_length
        mycopy.set_empirical_model_name(self._empirical_model_name)
        self.check_data_and_shared_vars_on_equality()
        mycopy.check_data_and_shared_vars_on_equality()
        mycopy._data_type_mapper = self._data_type_mapper.copy()
        mycopy.probabilistic_program_graph = cp.copy(self.probabilistic_program_graph)
        mycopy._update_samples_model_representation(recreate_samples_model_repr=False)
        return mycopy

    def _maximum(self):
        """Returns the point of the maximum density in this model"""
        row_cnt, col_cnt = self.samples.shape
        if row_cnt == 0:
            # can not compute any aggregation. return nan
            return [None] * col_cnt
        # Set starting point for optimization problem
        opt = sciopt.minimize(self._negdensity, self._samples_model_repr_mean, method='nelder-mead',
                              options={'xtol': 1e-8, 'disp': False})
        maximum = opt.x
        # Do not return a value if the density of the found maximum is NaN. In this case it is assumed that all
        # density values are NaN, so there should not be returned a maximum
        if np.isnan(opt.fun):
            return np.full(self.dim, np.nan)
        return maximum

    def check_data_and_shared_vars_on_equality(self):
        if self.shared_vars and not self.data.empty:
            for name in list(self.shared_vars.keys()):
                if name in self.data.columns:
                    assert np.array_equal(self.shared_vars[name].get_value(), np.array(self.data[name]))
