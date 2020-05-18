import logging
import numpy as np
import pandas as pd

from mb_modelbase.models_core import domains as dm
from mb_modelbase.models_core.base import Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


""" Utility functions for data import. """


def split_training_test_data(df, enabled=True):
    """Split data frame `df` into two parts and return them as a 2-tuple.

    The first returned data frame will contain 5% of the data, but not less than 25 item and not more than 50 items,
    and not more than 50% items.
    """
    if enabled:
        # select training and test data
        n = df.shape[0]
        limit = int(min(n * 0.10, 250, n*0.50))  # 10% of the data, but not more than 250 or 50%
        test_data = df.iloc[:limit, :]
        data = df.iloc[limit:, :]
    else:
        test_data = pd.DataFrame(columns=df.columns)
        data = df
    return test_data, data


def normalize_dataframe(df, numericals):
    """Normalizes all columns in data frame df. It uses z-score normalization and applies it per column. Returns the normalization parameters and the normalized dataframe,  as a tuple of (df, means, sigma). It expects only numercial columns in given dataframe.

    Args:
        df: dataframe to normalize.
    Returns:
        (df, means, sigmas): the normalized data frame, and the mean and sigma as np.ndarray
    """
    df = df.copy()
    numdf = df.loc[:, numericals]

    (n, dg) = numdf.shape
    means = numdf.sum(axis=0) / n
    sigmas = np.sqrt((numdf ** 2).sum(axis=0) / n - means ** 2)

    df.loc[:, numericals] = (numdf - means) / sigmas

    return df, means.values, sigmas.values


def clean_dataframe(df):
    # check that there are no NaNs or Nones
    if df.isnull().any().any():
        raise ValueError("DataFrame contains NaNs or Nulls.")

    # convert any categorical columns that have numbers into strings
    # and raise errors for unsupported dtypes
    for colname in df.columns:
        col = df[colname]
        dtype = col.dtype
        if dtype.name == 'category':
            # categories must have string levels
            cat_dtype = col.cat.categories.dtype
            if cat_dtype != 'str' and cat_dtype != 'object':
                logger.warning('Column "' + str(colname) +
                               '" is categorical, however the categories levels are not of type "str" or "object" '
                               'but of type "' + str(cat_dtype) +
                               '". I\'m converting the column to dtype "object" (i.e. strings)!')
                df[colname] = col.astype(str)

    return df


def get_columns_by_dtype(df):
    """Returns a triple of colnames (all, cat, num) where:
      * all is all names of columns in df,
      * cat is the names of all categorical columns in df, and
      * num is the names of all numerical columns in df.
      Any column in df that is not recognized as either categorical or numerical will raise a TypeError.
      """
    all = []
    categoricals = []
    numericals = []
    for colname in df:
        column = df[colname]
        if column.dtype.name == "category" or column.dtype.name == "object":
            categoricals.append(colname)
        elif np.issubdtype(column.dtype, np.number):
            numericals.append(colname)
        else:
            raise TypeError("unsupported column dtype : " + str(column.dtype.name) + " of column " + str(colname))
        all.append(colname)
    return all, categoricals, numericals


def get_discrete_fields(df, colnames):
    """Returns discrete fields constructed from the columns in colname of dataframe df.
    This assumes colnames only contains names of discrete columns of df."""
    fields = []
    for colname in colnames:
        column = df[colname]
        domain = dm.DiscreteDomain()
        extent = dm.DiscreteDomain(sorted(column.unique()))
        field = Field(colname, domain, extent, False, 'string', 'observed')
        fields.append(field)
    return fields


def get_numerical_fields(df, colnames, extend_extent=True, extend_by=0.05):
    """Returns numerical fields constructed from the columns in colname of dataframe df.
    This assumes colnames only contains names of numerical columns of df. Also, since fields are
    constructed from a data frame the variables are assumes to be 'observed' and not latent.

    Args:
        colnames: list of string
        extend_extent: bool, optional. Defaults to False.
            If true extend the returned extent of fields by 5% of original extent at lower and
            high end.
        extend_by: float, optional. Defaults to 0.05.
            Only has effect if extend_extent is set to true. Determines the amount of extension in
            % of the original extent.
    """
    fields = []
    for colname in colnames:
        column = df[colname]
        mi, ma = column.min(), column.max()
        d = (ma - mi) * extend_by if extend_extent else 0
        field = Field(colname, dm.NumericDomain(), dm.NumericDomain(mi-d, ma+d),
                      False, 'numerical', 'observed')
        fields.append(field)
    return fields


def to_category_cols(df, colnames):
    """Returns df where all columns with names in colnames have been converted to the category type using pd.astype(
    'category').
    """
    # df.loc[:,colnames].apply(lambda c: c.astype('category'))  # also works, but more tedious merge with not converted df part
    for c in colnames:
        # .cat.codes access the integer codes that encode the actual categorical values. Here, however, we want such integer values.
        df[c] = df[c].astype('category').cat.codes
    return df


def to_string_cols(df, columns=None, inplace=False):
    """Replace columns with their string representation.
    :param df: pd.DataFrame
        The data frame to work on
    :param inpace: boolean, optional. Defaults to False.
        Modifies df instead of returning a copy of it
    :param columns: list of strings, optional. Defaults to df.columns.
    :return:
    """
    if columns is None:
        columns = df.columns
    if not inplace:
        df = df.copy()
    df.loc[:, columns] = df.loc[:,columns].applymap(str)
    return df


def to_binned_stringed_series(series, bins):
    """Bin numerical column into equidistant bins.
    :param series: pd.Series
        The series to bin
    :param bins: int > 0
    """

    return pd.cut(series, bins=bins, precision=2).apply(str)


if __name__ == '__main__':
    series = pd.Series(list(range(10)))
    print(to_binneded_string_series(series, bins=3))
