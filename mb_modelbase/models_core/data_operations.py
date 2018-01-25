import numpy as np
import pandas as pd

from mb_modelbase.utils import utils


def condition_data(df, where):
    # make it a sequence if not already
    if isinstance(where, tuple):
        where = [where,]
    # iteratively build boolean selection mask
    mask = [True] * len(df)
    for (name, operator, values) in where:
        operator = operator.lower()
        column = df[name]
        if operator == 'in':
            if column.dtype == 'object':  # categorical column
                # df = df.loc[column.isin(values)]
                mask &= column.isin(values)
            else:  # quantitative column
                mask &= column.between(*values, inclusive=True)
        else:
            # values is necessarily a single scalar value, not a list
            if operator == 'equals' or operator == '==':
                # df = df.loc[column == values]
                mask &= column == values
            elif operator == 'greater' or operator == '>':
                # df = df.loc[column > values]
                mask &= column > values
            elif operator == 'less' or operator == '<':
                # df = df.loc[column < values]
                mask &= column < values
            else:
                raise ValueError('invalid operator for condition: ' + str(operator))
    return df.loc[mask, :]


def reduce_to_scalars(values):
    """Reduce all elements of values to scalars, as follows:
       * a scalar s are kept: s -> s
       * an interval [a,b] is reduced to its mean: [a,b] -> (a+b)/2
       Note that it only accepts numbers or intervals of numbers.
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
