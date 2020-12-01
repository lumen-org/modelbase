import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"


# tars1: width of the first joint of the first tarsus in microns
# tars2: _____________second_____________first__________________
# head: the maximal width of the head between the external edges of the exes in 0.01mm
# aede1: the maximal wodth of the aedeagus in the fore-part in microns
# aede2: the front angle of the aedeagus (1unit = 7.5 degree)
# aede3: the aedeagus width fromthe side in microns


def mixed(filepath=_csvfilepath):
    df = pd.read_csv(filepath)
    return df


if __name__ == '__main__':
    df = mixed()
    print(df.head())
