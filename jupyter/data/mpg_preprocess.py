"""
@author: Philipp Lucas
This data set is taken from: https://github.com/hadley/data-fuel-economy
"""

import pandas as pd

# that should about be all useful variables:
_cols = ['year', 'car_size', 'displacement', 'cylinder', 'mpg_city', 'mpg_highway', 'drv',
         'transmission', 'turbo']

# temporal variables:
_temp_cols = ['year']
# year is first year when car has been sold

# numerical variables
_num_cols = ['displacement', 'cylinder', 'mpg_city', 'mpg_highway']
# displ is displacement of engine
# cyl is number of cylinders
# cty is miles per gallon in city
# hwy is miles per gallon on highway

# categorical variables:
_cat_cols = ['car_size', 'drv', 'transmission', 'turbo']
# class is ""compact car" and such
# drv is RWD or FWD
# trans is transmission type

_mpg_filepath = './mpg.csv'


def _read(file=_mpg_filepath):
    df = pd.read_csv(file)
    df = pd.DataFrame(df, columns=['trans', 'cyl', 'T', 'class', 'year', 'cty', 'hwy', 'displ'])
    df = df.rename(columns={'cty': 'mpg_city', 'hwy': 'mpg_highway', 'trans': 'transmission',
                            'T': 'turbo', 'cyl': 'cylinder', 'class': 'car_size',
                            'displ': 'displacement'})
    return df


def preprocess(file=_mpg_filepath, do_not_change_columns=['cylinder'], drop_columns=['turbo']):
    df = _read(file)
    df = pd.DataFrame(df,
                      columns=['transmission', 'cylinder', 'turbo', 'car_size', 'year', 'mpg_city',
                               'mpg_highway', 'displacement'])

    df.car_size.replace(to_replace={'pickup': 'large', 'suv': 'midsize', 'station wagon': 'midsize',
                                    'compact': 'small', 'passenger van': 'large',
                                    'cargo van': 'large', 'two seater': 'small',
                                    'large car': 'large', 'midsize car': 'midsize',
                                    'compact car': 'small'}, inplace=True)

    if 'cylinder' not in do_not_change_columns:
        df.cylinder.replace(to_replace={2: 'few', 3: 'few', 4: 'few', 5: 'medium', 6: 'medium',
                                        7: 'medium', 8: 'medium', 10: 'many', 12: 'many',
                                        16: 'many'}, inplace=True)

    df.replace(to_replace={'transmission': {
        '.*auto.*': 'auto'}}, inplace=True, regex=True)

    df.replace(to_replace={'transmission': {
        'lock-up.*': 'lock-up'}}, inplace=True, regex=True)

    df.replace(to_replace={'transmission': {
        'manual.*': 'manual'}}, inplace=True, regex=True)

    df = df.drop(df[df.transmission == 'semi-auto'].index)
    df = df.drop(df[df.transmission == 'creeper(C5)'].index)
    df = df.drop(df[df.car_size == 'spv'].index)
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)

    for col in ['transmission', 'cylinder', 'turbo', 'car_size']:
        if col not in do_not_change_columns:
            df[col] = df[col].astype('str')

    for col in drop_columns:
        del df['turbo']

    return df


if __name__ == '__main__':
    """Running this file recreates the content of mpg_clean.csv. For convenience mpg_clean.csv is
        already part of the repository.
    """
    df = preprocess(do_not_change_columns=[], drop_columns=[])
    df.head()
    df.to_csv('./mpg_clean.csv', index=False)
    pass