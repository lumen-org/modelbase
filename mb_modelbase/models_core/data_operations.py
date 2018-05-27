# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)


def condition_data(df, where):
    """ Conditions data frame df according to conditions in where and returns a view on the remaining, filtered data frame.

    :param df: A data frame.
    :param where: a single condition, or a sequence of conditions. A condition is a three-tuple of (name, operator, values).
    :return: pd.DataFrame
    """
    # make it a sequence if not already
    if isinstance(where, tuple):
        where = [where,]
    # iteratively build boolean selection mask
    mask = [True] * len(df)
    for (name, operator, values) in where:
        operator = operator.lower()
        column = df[name]
        if operator == 'in':
            if column.dtype == 'object' or column.dtype.name == 'category':  # categorical column
                # df = df.loc[column.isin(values)]
                mask &= column.isin(values)
            else:  # quantitative column
                mask &= column.between(*values, inclusive=True)  # TODO: inclusive??
        else:
            # values is necessarily a single scalar value, not a list
            if operator == 'equals' or operator == '==':
                # df = df.loc[column == values]
                mask &= column == values
            elif operator == 'greater' or operator == '>':
                # df = df.loc[column > values]
                mask &= column >= values  # TODO: i use >= !!
            elif operator == 'less' or operator == '<':
                # df = df.loc[column < values]
                mask &= column < values
            else:
                raise ValueError('invalid operator for condition: ' + str(operator))
    # apply mask all at once
    return df.loc[mask, :]


def density(df, x):
    """Returns the density (i.e. absolute frequency) of point x in DataFrame df.

    Args:
        df: A data frame.
        x: a sequence of values of the dimensions of DataFrame df, in the same order as in df.columns.
    """
    dim = df.shape[1]
    names = df.columns
    # TODO: count matches instead of data frame construction? should be faster
    reduced_df = condition_data(df, zip(names, ['=='] * dim, x))
    return reduced_df.shape[0]


def probability(df, domains):
    """Returns the probability (i.e relative, empirical frequency) of the space described by

     Args:
        df: A data frame.
        domains: a sequence of domains of the dimensions of DataFrame df, in the same order as in df.columns.
            A domain may not be a scalar value, but must be a sequence. Even if it only holds one element.
     """
    if df.shape[0] == 0:
        return 0
    dim = df.shape[1]
    names = df.columns
    # TODO: count matches instead of data frame construction? should be faster
    reduced_df = condition_data(df, zip(names, ['in'] * dim, domains))
    return reduced_df.shape[0] / df.shape[0]


def reduce_to_scalars(values):
    """Reduce all elements of values to scalars, as follows:
       * a scalar s are kept: s -> s
       * an interval [a,b] is reduced to its mean: [a,b] -> (a+b)/2
       NOTE: it only accepts numbers or intervals of numbers.
    """
    v = []
    for value in values:
        # all of the todos in filter apply here as well...
        try:
            if not isinstance(value, str):
                v.append((value[0] + value[1]) / 2)
            else:
                v.append(value)
        except (TypeError, IndexError):
            v.append(value)
    return v

def scalarize(lst):
    """Given a list of any mix of scalars (string or numbers) and single-element lists of scalars return a flattened list, i.e. a list of scalars. It is stable, i.e. the order of elements is kept."""
    pass
