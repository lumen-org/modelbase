import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"

def mixed(filepath=_csvfilepath):

    df = pd.read_csv(filepath, index_col=0)

    del df['lived_abroad']
    del df['happiness']
    del df['health']

    return df