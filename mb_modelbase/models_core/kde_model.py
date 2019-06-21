# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)

from mb_modelbase.models_core.models import Model
from mb_modelbase.models_core import data_operations as data_op
from mb_modelbase.models_core import data_aggregation as data_aggr
from scipy import stats
from mb_modelbase.utils import data_import_utils
import copy
import numpy as np
import pandas as pd
import scipy.optimize as sciopt
import itertools


class KDEModel(Model):
    """
    A Kernel Density Estimator (KDE) model is a model whose distribution is determined by using a kernel
    density estimator. KDE work in a way that to each point of the observed data a so-called kernel distribution is
    assigned centered at that point (e.g. a normal distribution). The distributions from all data points
    are then summed up and build up the joint distribution for the model
    (see https://scikit-learn.org/stable/modules/density.html#kernel-density)
    TODOs:
     - model with categorical variables are not yet supported. The standard scikit kernel density estimator used here
       only supports continuous data
     - Calculating the point of maximum density is very slow, since a number of optimization problems is solved that
       quadratically increases with the number of dimensions. Try different approaches: Maybe guess one starting point
       and then only solve one optimization problem
     - bandwidth parameter for the kernel density estimation is currently always set to 0.1. I couldN#t find a
       heuristic for multidemsnional models. A good value for the bandwidth probably has to be calculated dynamically
    """

    def __init__(self, name):
        super().__init__(name)
        self.kde = None
        self._emp_data = None
        self._aggrMethods = {
            'maximum': self._maximum,
        }
        self.parallel_processing = False

    def _set_data(self, df, drop_silently, **kwargs):
        #assert data_import_utils.get_columns_by_dtype(df)[1] == [], \
        #    'kernel density estimation is possible only for continuous data'
        self._set_data_mixed(df, drop_silently, split_data=False)
        self.test_data = self.data.iloc[0:0, :]
        return ()

    def _fit(self):
        # Split data into numerical and categorical variables
        num_idx = []
        for idx, dtype in enumerate(self.data.dtypes):
            if np.issubdtype(dtype, np.number):
                num_idx.append(idx)
        #if num_idx:
            # Perform kernel density estimation for numerical dimensions
            #self.kde = stats.gaussian_kde(self.data.iloc[:, num_idx].T)
            # This is necessary for conditioning on the data later
        self._emp_data = self.data.copy()
        return()

    def _conditionout(self, keep, remove):
        """Conditions the random variables with name in remove on their domain and marginalizes them out.
        """
        for name in remove:
            # Remove all rows from the _emp_data that are outside the domain in name
            lower_bound = self.byname(name)['domain'].values()[0]
            upper_bound = self.byname(name)['domain'].values()[1]
            lower_cond = self._emp_data[name] >= lower_bound
            upper_cond = self._emp_data[name] <= upper_bound
            self._emp_data = self._emp_data[lower_cond & upper_cond]

        # transfer condition from _emp_data to data
        self.data = self.data.loc[list(self._emp_data.index.values), :]
        self._marginalizeout(keep, remove)
        return()

    def _marginalizeout(self, keep, remove):
        """Marginalizes the dimensions in remove, keeping all those in keep"""
        # Fit the model to the current data. The data dimension to marginalize over
        # should have been removed before
        self._fit()
        _, cat_names, num_names = data_import_utils.get_columns_by_dtype(self.data)
        self._categoricals = cat_names
        self._numericals = num_names
        return ()

    def get_relative_frequency(self, value):
        value = list(value)
        occurence = [row == value for row in self.data.values.tolist()]
        freq = sum(occurence)
        rel_freq = freq / self.data.shape[0]
        return rel_freq

    def _density(self, x):
        """Returns the density at x"""
        # Sort x in the same way as the data
        x_df = pd.DataFrame([x])
        _, cat_names, num_names = data_import_utils.get_columns_by_dtype(x_df)
        x_df = x_df[cat_names + num_names]
        x = x_df.values.tolist()[0]
        if not num_names:
            return self.get_relative_frequency(x)
        else:
            # Split data into numerical and categorical variables
            num_idx = []
            cat_idx = []
            for idx, dtype in enumerate(self.data.dtypes):
                if np.issubdtype(dtype, np.number):
                    num_idx.append(idx)
                else:
                    cat_idx.append(idx)
            x_num = [x[i] for i in num_idx]
            x_cat = [x[i] for i in cat_idx]
            # Condition numeric data on categorical data
            m = self.copy()
            for i in cat_idx:
                m.data = m.data[m.data.iloc[:, i] == x[i]]
            # Get density of conditioned model p(num|cat)
            m.kde = stats.gaussian_kde(m.data.iloc[:, num_idx].T)
            x_num = np.reshape(x_num, (1, len(x_num)))
            cond_density = m.kde.evaluate(x_num)
            # Get marginal density of categorical variables p(cat)
            cat_density = len(m.data)/len(self.data)
            # p(num,cat) = p(num|cat) * p(cat)
            density = cond_density * cat_density
            return density[0]

    def _negdensity(self, x):
        return -self._density(x)

    def _maximum(self):
        """Compute the point of maximum density """
        # # The problem is not convex, so try different starting points and return the biggest of the found maxima
        #
        # # Generate starting points
        # nr_of_points_per_dim = 20
        # total_nr_of_points = nr_of_points_per_dim ** len(self.fields)
        # max_vals = [self.fields[i]['extent'].values()[1] for i in range(len(self.fields))]
        # min_vals = [self.fields[i]['extent'].values()[0] for i in range(len(self.fields))]
        # extent_range = [x_max - x_min for x_max, x_min in zip(max_vals, min_vals)]
        # stepsize = [i/nr_of_points_per_dim for i in extent_range]
        # stepsize_re = np.reshape(stepsize, (len(stepsize), 1))
        # points_re = np.reshape(np.arange(nr_of_points_per_dim), (1, nr_of_points_per_dim))
        # min_vals_re = np.reshape(min_vals, (len(self.fields), 1))
        # points_per_dim = np.inner(stepsize_re, points_re.T) + min_vals_re
        # # Perform cross join
        # starting_points = np.array(np.meshgrid(*points_per_dim)).T.reshape(total_nr_of_points, len(self.fields))


        # Get all combinations of categorical values
        unique_vals = [self.fields[i]['extent'].values() for i in range(len(self._categoricals))]
        cartesian_prod = list(itertools.product(*unique_vals))
        if self._numericals:
            # Solve an optimization problem for each of the values
            global_max_num = [np.mean(self.data[col]) for col in self._numericals]
            global_max_density = 0
            for cat_val in cartesian_prod:
                m = self.copy()
                # Condition on categorical values
                for i, val in enumerate(cat_val):
                    m.data = m.data[m.data.iloc[:, i] == val]
                # kde can't handle datasets with only one row
                if m.data.shape[0] < 2:
                    continue
                m.marginalize(m._numericals)
                # Solve the optimization problem
                x0 = [np.mean(m.data[col]) for col in m._numericals]
                local_max = sciopt.minimize(m._negdensity, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
                local_max_cond_density = -local_max.fun
                local_max_marginal_density = len(m.data)/len(self.data)
                local_max_joint_density = local_max_cond_density * local_max_marginal_density
                if local_max_joint_density > global_max_density:
                    global_max_density = local_max_joint_density
                    global_max_num = local_max.x.tolist()
                    global_max_cat = list(cat_val)
            return global_max_cat + global_max_num
        else:
            densities = [self.get_relative_frequency(i) for i in cartesian_prod]
            idx = densities.index(max(densities))
            return list(cartesian_prod[idx])

    def _arithmetic_mean(self):
        """Returns the point of the average density"""
        mean = data_aggr.aggregate_data(self.data, 'maximum')
        return mean

    def _sample(self):
        """Returns random point of the distribution"""
        sample = np.zeros(len(self.fields))
        # Split data into numerical and categorical variables
        num_idx = []
        cat_idx = []
        for idx, dtype in enumerate(self.data.dtypes):
            if np.issubdtype(dtype, np.number):
                num_idx.append(idx)
            else:
                cat_idx.append(idx)
        # get samples for numerical dimensions
        sample[num_idx]= self.kde.sample()
        # get samples for categorical dimensions
        sample[cat_idx] = self.data.iloc[:, cat_idx].sample().tolist() #[0] ?
        return sample

    def copy(self, name=None):
        mycopy = self._defaultcopy(name)
        mycopy.kde = copy.deepcopy(self.kde)
        mycopy._emp_data = self._emp_data.copy()
        mycopy._categoricals = self._categoricals.copy()
        mycopy._numericals = self._numericals.copy()
        return mycopy


# Create model for testing
if __name__ == "__main__":
    import pandas as pd
    import mb_modelbase as mb

    # size = 10000
    # a1 = np.random.normal(6, 3, int(size*0.75))
    # a2 = np.random.normal(1, 0.5, int(size*0.25))
    # a = np.concatenate((a1, a2))
    # np.random.shuffle(a)
    # b = np.random.normal(0, 1, size)
    # data = pd.DataFrame({'A': a, 'B': b})
    # kde_model = mb.KDEModel('kde_model_multimodel_gaussian')

    # data = pd.read_csv('/home/guet_jn/Desktop/mb_data/mb_data/iris/iris_numeric.csv')
    # kde_model = mb.KDEModel('kde_model_iris')

    # data = pd.read_csv('/home/guet_jn/Desktop/mb_data/mb_data/mpg/mpg_numeric.csv')
    # kde_model = mb.KDEModel('kde_model_mpg')

    data = pd.read_csv('/home/guet_jn/Desktop/mb_data/mb_data/crabs/crabs_numeric.csv')
    kde_model = mb.KDEModel('kde_model_crabs')

    kde_model.fit(data)
    Model.save(kde_model, '/home/guet_jn/Desktop/mb_data/data_models/kde_model_crabs.mdl')


