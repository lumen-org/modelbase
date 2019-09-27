import numpy as np

TestQuantities = {
    """ Map of string ids of test quantities to the respective methods."""
    'average': lambda x: np.average(x, axis=0),
    'median': lambda x: np.median(x, axis=0),
    'variance': lambda x: np.var(x, axis=0),
    'min': lambda x: np.min(x, axis=0),
    'max': lambda x: np.max(x, axis=0),
}


def posterior_predictive_check(model, test_quantity_fct, k=None, n=None, reference_data=None):
    """ Do a posterior predictive check.

    Args
        model: Model
            The model to use for the ppc.
        test_quantity_fct: callable, string
            The test quantity to use. The test quantity function accept one or two dimensional arrays and compute the
            value of each column of the data. See also TestQuantities for examples.
        k: int
            Number of samples to draw each round. If not set the size of training data of model is used.
        n: int
            Number of rounds to draw k samples and compute test quantity on it. Defaults to 50
        reference_data: pd.DataFrame
            The reference data to compare to. If not set the models training data is used.

    Return: (np.array, np.array)
        2-tuple of test quantity value for each field of the model, and np.array of test quantity value for the k sets
        for each field of the model
    """
    if model.dim == 0:
        raise ValueError("cannot do posterior predictive check on zero-dimensional model.")

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
    samples = map(lambda _: model.sample(k).values, ks)
    samples_test = list(map(test_quantity_fct, samples))
    reference_test = test_quantity_fct(reference_data.values)

    # transpose such that first axis corresponds to variables and second to sample set index
    samples_test = np.asarray(samples_test).transpose()
    reference_test = np.asarray(reference_test).transpose()

    return reference_test, samples_test

