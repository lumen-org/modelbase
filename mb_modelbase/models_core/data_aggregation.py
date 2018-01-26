# copyright (c) 2017, Philipp Lucas, philipp.lucas@uni-jena.de
""" each function receives a pandas data frame
and returns a particular type of aggregation as a pandas series"""

import numpy as np
import pandas as pd

from mb_modelbase.utils import utils

DEFAULT_BIN_NUMBER = 10


def most_frequent_equi_sized(data, opts=None):
    """ Expects a pandas data frame of mixed (i.e. numerical and categorical) columns.
    It returns a most frequent item, with details as follows:

    'most frequent' is a difficult concept in context of continuous domains. One approach is to discretize the
      domain. Here we discretize each continuous domain by splitting it in opts[0] many disjunct intervals such that
      each intervals has the same length. The most frequent item then is the center of the interval that occurs
      the most.
    """

    n, d = data.shape
    k = opts[0] if (opts is not None and opts != []) else DEFAULT_BIN_NUMBER
    k = min(k, n)

    # derive interval levels for each numerical column
    # TODO: this _copies_ the whole data!!!
    # todo: can I circumvent this by passing in the numpy structures instead?

    numeric = [np.issubdtype(dtype, np.number) for dtype in list(data.dtype)]

    if any(numeric):
        df = pd.DataFrame()
        for idx, colname in enumerate(data.columns):
            if numeric[idx]:
                # attached leveled numerical column by cutting it to levels
                df[colname], bins = pd.cut(x=data[colname], bins=k, retbins=True)
                # change level values to the later result
                df[colname].cat.categories = utils.rolling_1d_mean(bins)
            else:
                df[colname] = data[colname]
    else:
        df = data

    # find observation with highest number of occurrences
    grps = df.groupby(list(df.columns))
    # TODO: allow Nans in the result! it fails on the client when decoding the JSON at the moment
    if len(grps) == 0:
        return [0] * d
    else:
        data_res = grps.size().argmax()
        #assert (len(data_res) == len(df.columns) and len(data_res) == d)
        return [data_res] if d == 1 else list(data_res)


def most_frequent_equi_massed(data, opts=None):
    """ Expects a pandas data frame of mixed (i.e. numerical and categorical) columns.
    It returns a most frequent item, with details as follows:

    'most frequent' is a difficult concept in context of continuous domains. One approach is to discretize the
      domain. Here we discretize each continuous domain by splitting it in opts[0] many disjunct intervals such that
      each intervals contains the same amount of points. The most frequent item then is the center of the interval
      that is the shortest.
    """

    raise NotImplementedError("this is bullshit")

    # TODO: this whole approach is bullshit! darn!
    # problem is: I split each continuous column into intervals that have the same number of occurances - and after that
    #  I count occurances. I really only works for a one-dimensional cont case ...
    # even for a pure continuous case I'm not sure anymore if that makes sense.  what I really would like to do is:
    #  split the all cont dimensions together into subspaces that each contain equally many items.
    # Therefore: this approach only makes sense for a one dimensional cont case...

    n, d = data.shape

    k = opts[0] if (opts is not None and opts != []) else DEFAULT_BIN_NUMBER
    k = min(k, len(data))

    # derive interval levels for each numerical column
    # TODO: this _copies_ the whole data!!!
    # todo: can I circumvent this by passing in the numpy structures instead?
    mycopy = pd.DataFrame()
    for colname in data.columns:
        dtype = data[colname].dtype.name
        if dtype == "category" or dtype == "object":
            mycopy[colname] = data[colname]
        else:
            # attached leveled numerical column
            # TODO: ahhh!! stupid me. I could have used pd.qcut!
            bins = utils.equiweightedintervals(seq=data[colname].tolist(), k=k, bins=True)
            # collapse to unique bins
            bins = sorted(set(bins))
            # turn bins to rolling means and use that as levels - this way we dont need to convert the labels back later
            labels = utils.rolling_1d_mean(bins)
            # cut to levels
            mycopy[colname] = pd.cut(x=data[colname], bins=bins, include_lowest=True, labels=labels)

    # find observation with highest number of occurrences
    allcols = list(mycopy.columns)
    grps = mycopy.groupby(allcols)

    # TODO: allow Nans in the result! it fails on the client when decoding the JSON at the moment
    if len(grps) == 0:
        return [0] * d
    else:
        data_res = grps.size().argmax()
        return [data_res] if d == 1 else list(data_res)


def most_frequent(df):
    """ Expects a pandas data frame of only categorical columns (dtype == 'object' or dtype == 'string').
    Returns the most frequent row in the data frame.
    """
    assert(np.number not in list(df.dtypes))

    n, d = df.shape
    allcols = list(df.columns)
    grps = df.groupby(allcols)

    # TODO: allow Nans in the result! it fails on the client when decoding the JSON at the moment
    if len(grps) == 0:
        return [0] * n
    else:
        data_res = grps.size().idxmax()
        return [data_res] if d == 1 else list(data_res)


def average_most_frequent(df, opts=None):
    """ Expects a pandas data frame of possibly both numerical and categorical columns.
    Returns an 'mixed-heuristic' average-maximum aggregation, as follows:
        For the categorical part of the data it returns the most frequent row
        For the numerical part of the data it returns the average row
    """
    n,d = df.shape
    if n == 0:
        raise NotImplementedError("cannot return NaNs at the moment.")
    if d == 0:
        raise ValueError("cannot aggregate dataframe without any column")

    num_idx = []
    cat_idx = []

    for idx, dtype in enumerate(df.dtypes):
        if np.issubdtype(dtype, np.number):
            num_idx.append(idx)
        else:
            cat_idx.append(idx)

    if len(num_idx) == 0:
        return most_frequent(df)
    if len(cat_idx) == 0:
        return average(df)
    else:
        num_avg = average(df.iloc[:, num_idx])
        cat_mstfrqt = most_frequent(df.iloc[:, cat_idx])
        return utils.mergebyidx(num_avg, cat_mstfrqt, num_idx, cat_idx)


def average(df):
    """ Expects a pandas data frame of only numeric columns (anything derived from numpy.number).
    Returns the average data item in df as an array.
    """
    assert(df.select_dtypes(include=[np.number]).shape[1] == df.shape[1])
    return df.mean(axis=0).values  # values always returns an array


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import data.crabs.crabs as crabs
    import data.iris.iris as iris_

    iris = iris_.mixed('data/iris/iris.csv')
    crabs = crabs.mixed('data/crabs/australian-crabs.csv')

    print('average iris cont: ')
    print(average(iris.iloc[:, :-1]))

    print('average crabs cont: ')
    print(average(crabs.iloc[:, 2:]))

    # print('most_frequent_equi_massed iris mixed')
    # print(most_frequent_equi_massed(iris, opts=[5]))
    #
    # print('most_frequent_equi_massed crabs mixed')
    # print(most_frequent_equi_massed(crabs, opts=[5]))

    print('most_frequent_equi_sized iris mixed')
    print(most_frequent_equi_sized(iris, opts=[5]))

    print('most_frequent_equi_sized crabs mixed')
    print(most_frequent_equi_sized(crabs, opts=[5]))

    print('average_most_frequent iris mixed')
    print(average_most_frequent(iris))

    print('average_most_frequent crabs mixed')
    print(average_most_frequent(crabs))
