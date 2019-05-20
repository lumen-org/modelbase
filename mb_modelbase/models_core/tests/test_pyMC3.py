import numpy as np
import pandas as pd
import mb_modelbase as mbase
import unittest

model_paths = [
               '/home/guet_jn/Desktop/mb_data/data_models/pymc3_getting_started_model.mdl',
               '/home/guet_jn/Desktop/mb_data/data_models/pymc3_simplest_model.mdl',
               '/home/guet_jn/Desktop/mb_data/data_models/pymc3_coal_mining_disaster_model.mdl',
               '/home/guet_jn/Desktop/mb_data/data_models/eight_schools_model.mdl',
               '/home/guet_jn/Desktop/mb_data/data_models/pymc3_getting_started_model_independent_vars.mdl'
               ]

data_paths = [
              '/home/guet_jn/Desktop/mb_data/mb_data/pymc3/getting_started.csv',
              '/home/guet_jn/Desktop/mb_data/mb_data/pymc3/simplest_testcase.csv',
              '/home/guet_jn/Desktop/mb_data/mb_data/pymc3/coal_mining_disasters.csv',
              '/home/guet_jn/Desktop/mb_data/mb_data/pymc3/eight_schools.csv',
              '/home/guet_jn/Desktop/mb_data/mb_data/pymc3/pymc3_getting_started_model_independent_vars.csv'
             ]

class Test_methods_on_initialized_model(unittest.TestCase):
    """
    Test the ProbabilisticPymc3Model methods on a model that has just been initialized
    """

    def testinit(self):
        """
        Test if newly initialized model has data, test data or samples and if mode is set to None
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            self.assertEqual(mymod.data.empty, 1, "There should be no data. Model:" + mymod.name)
            self.assertEqual(mymod.test_data.empty, 1, "There should be no test data. Model:" + mymod.name)
            self.assertEqual(mymod.samples.empty, 1, "There should be no samples. Model:" + mymod.name)
            self.assertIsNone(mymod.mode, "Mode of just instantiated model should be set to None. Model:" + mymod.name)

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.set_data(data)
            self.assertEqual(mymod.data.empty, 0, "There is no data in the model. Model: " + mymod.name)
            self.assertEqual(mymod.data.columns.equals(data.columns), 1,
                             "model data has different columns than the original data. Model: " + mymod.name)
            self.assertEqual(mymod.mode, 'data', "model mode should be set to data. Model: " + mymod.name)

    # TODO: What should happen, if the fit method is called on a model without data?
    # def test_fit(self):
    #     """
    #     Test if there are samples and test data in the model and if the mode is set to model
    #     """
    #     mymod = mbase.Model.load(self.testcasemodel_path)
    #     mymod.fit()
    #     self.assertEqual(mymod.samples.empty, 0, "There are no samples in the model")
    #     self.assertEqual(mymod.test_data.empty, 0, "There is no test data in the model")
    #     self.assertEqual(mymod.mode, 'model', "mode should be set to model")

    def test_marginalizeout(self):
        """
        Call _marginalizeout on a model without any samples.
        An error should be thrown since the model does not yet know any variables
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            with self.assertRaises(ValueError):
                mymod.density([0])

    def test_maximum(self):
        """
        Calculate the maximum probability of a model without samples. It should return an empty array
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            self.assertTrue(len(mymod._maximum()) == 0,
                            "maximum density point for a model without variables should be an empty array. "
                            "Model: " + mymod.name)

class Test_methods_on_model_with_data(unittest.TestCase):
    """
    Test the ProbabilisticPymc3Model methods on a model that has been initialized and given data
    """

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.set_data(data)
            with self.assertRaises(ValueError):
                mymod._conditionout(keep='A', remove='B')
            self.assertEqual(mymod.data.empty, 0, "There should be data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.empty, 1, "There should be no samples. Model: " + mymod.name)
            self.assertEqual(mymod.mode, "data",
                             "Mode of just instantiated model should be set to data. Model: " + mymod.name)

    def test_density(self):
        """
        Calculate a probability density on a model.
        An error should be thrown since the model does not yet know any variables
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.set_data(data)
            with self.assertRaises(ValueError):
                mymod.density([0])

    def test_maximum(self):
        """
        Calculate the maximum probability of a model without samples. It should return an empty array
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.set_data(data)
            self.assertTrue(len(mymod._maximum()) == 0,
                            "maximum density point for a model without samples should be an empty array. "
                            "Model: " + mymod.name)

class Test_methods_on_fitted_model(unittest.TestCase):
    """
    Test the ProbabilisticPymc3Model methods on a model that has been initialized, given data and fitted
    """

    def testcopy(self):
        """
        Test if data, test data and samples of the copied model are the same as in the original model
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.fit(data)
            mymod_copy = mymod.copy()
            self.assertEqual(mymod.data.equals(mymod_copy.data), 1,
                             "Copied model data is different than original model data. Model: " + mymod.name)
            self.assertEqual(mymod.test_data.equals(mymod_copy.test_data), 1,
                             "Copied model test data is different than original model test data. Model: " + mymod.name)
            self.assertEqual(mymod.samples.equals(mymod_copy.samples), 1,
                             "Copied samples are different than original samples. Model: " + mymod.name)

    def test_marginalizeout(self):
        """
        Call _marginalizeout on a fitted model. Check if the correct variables are removed from the model
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.fit(data)
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.fit(data)
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.fit(data)
            location = np.zeros(len(mymod.names))
            self.assertTrue(isinstance(mymod.density(location), float),
                            "A single scalar should be returned. Model: " + mymod.name)

    def test_maximum(self):
        """
        Calculate the maximum probability of a model. Dimensions should match
        """
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.fit(data)
            self.assertEqual(len(mymod._maximum()), len(mymod.names),
                             "Dimension of the maximum does not match dimension of the model. Model: " + mymod.name)

class Test_more_combinations_on_model(unittest.TestCase):
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
        for i, name in enumerate(model_paths):
            mymod = mbase.Model.load(model_paths[i])
            data = pd.read_csv(data_paths[i])
            mymod.fit(data)
            remove = mymod.names[0]
            mymod.marginalize(remove=remove)
            self.assertEqual(len(mymod._maximum()), len(mymod.names),
                             "Dimensions of the maximum and the model variables do not match. Model: " + mymod.name)


if __name__ == "__main__":

    unittest.main()









