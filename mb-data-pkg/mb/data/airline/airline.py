import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + "DelayDataProcessed.csv"


def data(file=_csvfilepath):
    df = pd.read_csv(file)
    df.dropna(axis=0, inplace=True)
    return df


if __name__ == "__main__":
    df = data()
