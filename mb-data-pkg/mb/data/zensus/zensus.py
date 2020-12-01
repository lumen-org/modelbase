import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"


def mixed(path=_csvfilepath):
    df = pd.read_csv(path, delimiter=';')
    df['Bundesland'] = df['Bundesland'].astype('category')
    df['Landkreis'] = df['Landkreis'].astype('category')
    return df

if __name__ == '__main__':
    df = mixed()