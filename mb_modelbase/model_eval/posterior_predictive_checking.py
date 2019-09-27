import numpy as np

TestQuantities = {
    """ Map of string ids of test quantities to the respective methods."""
    'average': np.average,
    'median': np.median,
    'variance': np.var,
    'min': np.min,
    'max': np.max,
}


def posterior_predictive_check(model, test_quantity_fct, k=None, n=None, reference_data=None):
    """ Do a posterior predictive check.

    Args
        model: Model
            The model to use for the ppc
        test_quantity_fct: callable, string
            The test quantity to use.
        k: int
            Number of samples to draw each round. If not set the size of training data of model is used.
        n: int
            Number of rounds to draw k samples and compute test quantity on it. Defaults to 50
        reference_data: pd.DataFrame
            The reference data to compare to. If not set the models training data is used.

    Return: (number, list(number))
        2-tuple of test quantity value of reference data and a list of the test quantity value for the k sets of
         samples.
    """
    if reference_data is None:
        reference_data = model.data
        if k is None:
            k = reference_data.shape[0]

    if k <= 0:
        raise ValueError("k must be a positive number > 0.")

    if n is None:
        n = 50
    if n <= 0:
        raise ValueError("n must be a positive number > 0.")

    ks = np.empty(n)
    ks.fill(k)
    samples = map(lambda k: model.sample(k).values, ks)
    samples_test = list(map(test_quantity_fct, samples))
    reference_test = test_quantity_fct(reference_data.values)

    return reference_test, samples_test

