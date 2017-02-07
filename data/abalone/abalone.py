import pandas as pd


def cg(file='data/abalone/abalone.csv'):
    df = pd.read_csv(file)
    df.dropna(axis=0, inplace=True)
    return df


if __name__ == "__main__":
    df = cg("abalone.csv")
