import os

_csvfilepath = os.path.splitext(__file__)[0] + ".data"


def data(filepath=_csvfilepath):
    raise Exception("not implemented")


if __name__ == '__main__':
    df = data()
    print(df.head())
