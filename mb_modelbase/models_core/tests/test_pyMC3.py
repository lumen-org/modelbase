import numpy as np
import unittest
import mb_modelbase.models_core.tests.create_PyMC3_testmodels as cr
import copy

<<<<<<< HEAD
model_paths = [
               '/home/guet_jn/Desktop/mb_data/data_models/pymc3_getting_started_model.mdl',
               #'/home/guet_jn/Desktop/mb_data/data_models/pymc3_simplest_model.mdl',
               #'/home/guet_jn/Desktop/mb_data/data_models/pymc3_coal_mining_disaster_model.mdl',
               #'/home/guet_jn/Desktop/mb_data/data_models/eight_schools_model.mdl',
               #'/home/guet_jn/Desktop/mb_data/data_models/pymc3_getting_started_model_independent_vars.mdl'
               ]

data_paths = [
              '/home/guet_jn/Desktop/mb_data/mb_data/pymc3/getting_started.csv',
              #'/home/guet_jn/Desktop/mb_data/mb_data/pymc3/simplest_testcase.csv',
              #'/home/guet_jn/Desktop/mb_data/mb_data/pymc3/coal_mining_disasters.csv',
              #'/home/guet_jn/Desktop/mb_data/mb_data/pymc3/eight_schools.csv',
              #'/home/guet_jn/Desktop/mb_data/mb_data/pymc3/pymc3_getting_started_model_independent_vars.csv'
             ]

class Test_methods_on_initialized_model(unittest.TestCase):
=======
def create_testmodels(fit):
    models = []
    # These functions return the model data and the corresponding model
    models.append(cr.create_pymc3_simplest_model(fit=fit))
    models.append(cr.create_pymc3_getting_started_model(fit=fit))
    models.append(cr.create_pymc3_getting_started_model_independent_vars(fit=fit))
    models.append(cr.create_pymc3_coal_mining_disaster_model(fit=fit))
    models.append(cr.create_getting_started_model_shape(fit=fit))
    models.append(cr.create_flight_delay_model(fit=fit))
    return models

#models_unfitted = create_testmodels(fit=False)
#models_fitted = create_testmodels(fit=True)



class TestMethodsOnInitializedModel(unittest.TestCase):
>>>>>>> master
    """
    Test the ProbabilisticPymc3Model methods on a model that has just been initialized
    """

    def testinit(self):
        """
        Test if newly initialized model has data, test data or samples and if mode is set to None
        """
        for data, mymod in create_testmodels(fit=False):
            self.assertEqual(mymod.data.empty, 1, "There should be no data. Model:" + mymod.name)
            self.assertEqual(mymod.test_data.empty, 1, "There should be no test data. Model:" + mymod.name)
            self.assertEqual(mymod.samples.empty, 1, "There should be no samples. Model:" + mymod.name)
            self.assertIsNone(mymod.mode, "Mode of just instantiated model should be set to None. Model:" + mymod.name)

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        for data, mymod in create_testmodels(fit=False):
            mymod_copy = mymod.copy()
            self.assertEqual(mymod.data.equals(mymod_copy.data), 1,
                             "Copied model data is different than original model data. Model:" + mymod.name)
            self.assertEqual(mymod.test_data.equals(mymod_copy.test_data), 1,
                             "Copied model test data is different than original model test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.equals(mymod_copy.samples), 1,
                             "Copied samples are different than original samples. Model: " + mymod.name)

    def test_set_data(self):
        """
        Test if the set_data() method gives any data to the model
        and if the model data has the same columns as the input data and if mode is set to data
        """
        for data, mymod in create_testmodels(fit=False):
            mymod.set_data(data)
            self.assertEqual(mymod.data.empty, 0, "There is no data in the model. Model: " + mymod.name)
            self.assertEqual(mymod.data.columns.equals(data.columns), 1,
                             "model data has different columns than the original data. Model: " + mymod.name)
            self.assertEqual(mymod.mode, 'data', "model mode should be set to data. Model: " + mymod.name)

    def test_marginalizeout(self):
        """
        Call _marginalizeout on a model without any samples.
        An error should be thrown since the model does not yet know any variables
        """
        for data, mymod in create_testmodels(fit=False):
            with self.assertRaises(ValueError):
                mymod._marginalizeout(keep='A', remove='B')
            self.assertEqual(mymod.data.empty, 1, "There should be no data. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.empty, 1, "There should be no test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 1, "There should be no samples. Model: " + mymod.name)
            self.assertIsNone(mymod.mode, "Mode of just instantiated model should be set to None. Model: " + mymod.name)

    def test_conditionout(self):
        """
        Call _conditionout on a model without any samples.
        An error should be thrown since the model does not yet know any variables
        """
        for data, mymod in create_testmodels(fit=False):
            with self.assertRaises(ValueError):
                mymod._conditionout(keep='A', remove='B')
            self.assertEqual(mymod.data.empty, 1, "There should be no data. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.empty, 1, "There should be no test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 1, "There should be no samples. Model: " + mymod.name)
            self.assertIsNone(mymod.mode, "Mode of just instantiated model should be set to None. Model: " + mymod.name)

    def test_density(self):
        """
        Calculate a probability density on a model.
        An error should be thrown since the model does not yet know any variables
        """
        for data, mymod in create_testmodels(fit=False):
            with self.assertRaises(ValueError):
                mymod.density([0])

    def test_maximum(self):
        """
        Calculate the maximum probability of a model without samples. It should return an empty array
        """
        for data, mymod in create_testmodels(fit=False):
            self.assertTrue(len(mymod._maximum()) == 0,
                            "maximum density point for a model without variables should be an empty array. "
                            "Model: " + mymod.name)


class TestMethodsOnModelWithData(unittest.TestCase):
    """
    Test the ProbabilisticPymc3Model methods on a model that has been initialized and given data
    """

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        for data, mymod in create_testmodels(fit=False):
            mymod.set_data(data)
            mymod_copy = mymod.copy()
            self.assertEqual(mymod.data.equals(mymod_copy.data), 1,
                             "Copied model data is different than original model data. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.equals(mymod_copy.test_data), 1,
                             "Copied model test data is different than original model test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.equals(mymod_copy.samples), 1,
                             "Copied samples are different than original samples. Model: " + mymod.name)

    def test_fit(self):
        """
        Test if there are samples, data and test data in the model and if the mode is set to both
        """
        for data, mymod in create_testmodels(fit=False):
            mymod.fit(data)
            self.assertEqual(mymod.data.empty, 0, "There is no data in the model. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.empty, 0, "There is no test data in the model. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 0, "There are no samples in the model. Model: " + mymod.name)
            self.assertEqual(mymod.mode, 'both', "mode should be set to both. Model: " + mymod.name)
            self.assertEqual(mymod.names, list(mymod.samples.columns.values),
                             "names and samples should hold the same variables in the same order. Model: " + mymod.name)
            self.assertEqual(mymod.names, [field['name'] for field in mymod.fields],
                             "names and fields should hold the same variables in the same order. Model: " + mymod.name)

    def test_marginalizeout(self):
        """
        Call _marginalizeout on a model without any samples for variables not in the model.
        An error should be thrown since the model does not have the variables
        """
        for data, mymod in create_testmodels(fit=False):
            mymod.set_data(data)
            with self.assertRaises(ValueError):
                mymod._marginalizeout(keep='A', remove='B')
            self.assertEqual(mymod.data.empty, 0, "There should be data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 1, "There should be no samples. Model: " + mymod.name)
            self.assertEqual(mymod.mode, "data",
                             "Mode of just instantiated model should be set to data. Model: " + mymod.name)

    def test_conditionout(self):
        """
        Call _conditionout on a model without any samples for variables not in the model.
        An error should be thrown since the model does not have the variables
        """
        for data, mymod in create_testmodels(fit=False):
            mymod.set_data(data)
            with self.assertRaises(ValueError):
                mymod._conditionout(keep='A', remove='B')
            self.assertEqual(mymod.data.empty, 0, "There should be data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 1, "There should be no samples. Model: " + mymod.name)
            self.assertEqual(mymod.mode, "data",
                             "Mode of just instantiated model should be set to data. Model: " + mymod.name)
    def test_maximum(self):
        """
        Calculate the maximum probability of a model without samples. It should return an empty array
        """
        for data, mymod in create_testmodels(fit=False):
            mymod.set_data(data)
            self.assertTrue(len(mymod._maximum()) == 0,
                            "maximum density point for a model without samples should be an empty array. "
                            "Model: " + mymod.name)

class TestMethodsOnFittedModel(unittest.TestCase):
    """
    Test the ProbabilisticPymc3Model methods on a model that has been initialized, given data and fitted
    """

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        for data, mymod in create_testmodels(fit=True):
            mymod_copy = mymod.copy()
            self.assertEqual(mymod.data.equals(mymod_copy.data), 1,
                             "Copied model data is different than original model data. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.equals(mymod_copy.test_data), 1,
                             "Copied model test data is different than original model test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.equals(mymod_copy.samples), 1,
                             "Copied samples are different than original samples. Model: " + mymod.name)

    def testcopy_after_change(self):
        """ Test if copied attributes change when the original model is changed"""
        for data, mymod in create_testmodels(fit=True):
            mymod_copy = mymod.copy()
            #Test name
            old_name = mymod.name
            new_name = 'qwertzuiopü'
            mymod.name = new_name
            self.assertTrue(mymod_copy.name == old_name,
                            "Name of copy is affected by changes in original model. Model: " + mymod.name)

            #Test_test_data
            old_test_data = mymod.test_data.copy()
            new_test_data = mymod.test_data + 1
            mymod.test_data = new_test_data
            self.assertTrue(mymod_copy.test_data.equals(old_test_data),
                            "Test data of copy is affected by changes in original model. Model: " + mymod.name)
            #Test samples
            old_samples = mymod.samples.copy()
            new_samples = mymod._sample(50)
            mymod.samples = new_samples
            self.assertTrue(mymod_copy.samples.equals(old_samples),
                            "Samples of copy are affected by changes in original model. Model: " + mymod.name)

            #Test shared vars
            if mymod.shared_vars:
                for key, value in mymod.shared_vars.items():
                    old_shared_vars = value.get_value()
                    mymod._sample(10)
                    self.assertTrue(np.array_equal(mymod_copy.shared_vars[key].get_value(), old_shared_vars),
                                    "Shared variables of copy are affected by changes in original model. "
                                    "Model: " + mymod.name)

            #Test empiricial_model_name
            old_emp_name = mymod._empirical_model_name
            new_emp_name = 'qwertzuiopü'
            mymod.set_empirical_model_name(new_emp_name)
            self.assertTrue(mymod_copy._empirical_model_name == old_emp_name,
                            "Empricial model name of copy is affected by changes in original model. "
                            "Model: " + mymod.name)

    def test_shared_vars_propagation_in_copy(self):
        """
        Test if after copying a model and changing the independent variables of the copy,
        these changes are propagated to the model_structure
        """
        for data, mymod in create_testmodels(fit=True):
            if mymod.shared_vars:
                mymod_cp = mymod.copy()
                key = list(mymod_cp.shared_vars.keys())[0]
                mymod_cp.shared_vars[key].set_value([1, 2, 3, 4])
                # _sample should not work anymore since the variables have now different lengths
                with self.assertRaises(ValueError):
                    mymod_cp._sample(1)

    def test_marginalizeout(self):
        """
        Call _marginalizeout on a fitted model. Check if the correct variables are removed from the model
        """
        for data, mymod in create_testmodels(fit=True):
            keep = mymod.names[1:]
            remove = [mymod.names[0]]
            mymod._marginalizeout(keep=keep, remove=remove)
            self.assertEqual(mymod.data.empty, 0,"There should be data. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.empty, 0, "There should be test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 0, "There should be samples. Model: " + mymod.name)
            self.assertEqual(mymod.mode, "both",
                             "Mode of just instantiated model should be set to both. Model: " + mymod.name)
            self.assertFalse(remove[0] in mymod.samples.columns,
                             str(remove) + " should be marginalized out and not be present in the samples. "
                                           "Model: " + mymod.name)
            self.assertTrue(all([name in mymod.samples.columns for name in keep]),
                            str(keep) + "should be still present in the samples. Model: " + mymod.name)

    def test_conditionout(self):
        """
        Call _conditionout on a fitted model. Check if the correct variables are removed from the model
        and if all the samples are within the variable domain
        """
        for data, mymod in create_testmodels(fit=True):
            keep = mymod.names[1:]
            remove = [mymod.names[0]]

            sample_size_all_values = len(mymod.samples)
            mymod.fields[0]['domain'].setupperbound(np.mean(mymod.samples[remove[0]]))
            isBiggerThanUpperBound = mymod.samples[remove[0]] > np.mean(mymod.samples[remove[0]])
            big_samples = mymod.samples[remove[0]][isBiggerThanUpperBound]
            sample_size_big_values = len(big_samples)

            mymod._conditionout(keep=keep, remove=remove)
            self.assertEqual(mymod.data.empty, 0, "There should be data. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.empty, 0, "There should be test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 0, "There should be samples. Model: " + mymod.name)
            self.assertEqual(mymod.mode,"both",
                             "Mode of just instantiated model should be set to both. Model: " + mymod.name)
            self.assertFalse(remove in mymod.samples.columns.values,
                             str(remove) + " should be marginalized out and not be present in the samples. "
                                           "Model: " + mymod.name)
            self.assertTrue(all([name in mymod.samples.columns for name in keep]),
                            str(keep) + "should be still present in the samples. Model: " + mymod.name)
            sample_size_small_values = len(mymod.samples)
            self.assertEqual(sample_size_all_values-sample_size_big_values, sample_size_small_values,
                             "numbers of removed samples and kept samples do not add up to previous number of samples. "
                             "Model: " + mymod.name)

    def test_density(self):
        """
        Calculate a probability density on a model. A single scalar should be the return value
        """
        for data, mymod in create_testmodels(fit=True):
            location = np.zeros(len(mymod.names))
            self.assertTrue(isinstance(mymod.density(location), float),
                            "A single scalar should be returned. Model: " + mymod.name)

    def test_maximum(self):
        """
        Calculate the maximum probability of a model. Dimensions should match
        """
        for data, mymod in create_testmodels(fit=True):
            self.assertEqual(len(mymod._maximum()), len(mymod.names),
                             "Dimension of the maximum does not match dimension of the model. Model: " + mymod.name)

    def test_sample(self):
        """
        Test if _sample() return the correct dimensions
        """
        for data, mymod in create_testmodels(fit=True):
            n = 10
            self.assertEqual(mymod.sample(n).shape[0],  n, 'Number of samples is not correct')
            self.assertEqual(mymod.sample(n).shape[1], mymod.samples.shape[1],
                             'Number of variables returned by _sample() is not correct')
            if mymod.shared_vars:
                mymod.marginalize(remove=list(mymod.shared_vars.keys())[0])
                self.assertEqual(mymod.sample(n).shape[1], mymod.samples.shape[1],
                                 'Number of variables returned by _sample() after marginalizing is not correct')


class TestMoreCombinationsOnModel(unittest.TestCase):
    """
    Test more complex cases, with more combinations of methods being applied to a already fitted model
    """

    # More combinations of marginalization and conditionalization cannot be applied
    # to the simple model since it only has two variables
    # TODO: What happens when each variable is marginalized out?

    def test_maximum_marginalized(self):
        """
        Check if the density maximum of a marginalized model has the same dimensions as the model variables
        """
        for data, mymod in create_testmodels(fit=True):
            remove = mymod.names[0]
            mymod.marginalize(remove=remove)
            self.assertEqual(len(mymod._maximum()), len(mymod.names),
                             "Dimensions of the maximum and the model variables do not match. Model: " + mymod.name)


if __name__ == "__main__":
    unittest.main()
