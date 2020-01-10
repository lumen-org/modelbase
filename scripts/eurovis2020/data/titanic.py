"""
@author: Fanli Lin

Data preprocessing and cleansing for the titanic data set
"""

import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"


def _clean(filepath=_csvfilepath):
    df = pd.read_csv(filepath)
    #del df['PassengerId']  # consecutive passanger id
    #del df['Name']  # name of a passanger
    del df['boat']  # ticket number
    del df['has_cabin_number']  # ticket number
    return df


def mixed(filepath=_csvfilepath):
    """Loads the titanic data set from a csv file, removes the index column and returns the
 	remaining data as a pandas data frame
    """
    df = _clean(filepath)

    # quantitize Parch, SibSp?
    #del df['SibSp']  # number of siblings/spouses aboard
    #del df['Parch']  # number of parents/children aboard

    # possibly quantitize PClass, i.e. the passenger class?
    return df


def continuous(filepath=_csvfilepath):
    df = _clean(filepath)
    df = df.replace({
        'pclass': {'Third': 3, 'Second': 2, 'First': 1}
    })
    df = df[['pclass','age','SibSp','fare']]
    return df


if __name__ == '__main__':
    # generate preprocessed csv data files
    df = mixed()
    df.to_csv(_csvfilepath + "_mixed2.csv", index=False)

    # df = continuous()
    # df.to_csv(_csvfilepath + "_continuous.csv", index=False)