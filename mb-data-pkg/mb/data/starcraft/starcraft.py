import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"


def cg(file=_csvfilepath):
    sc2df = pd.read_csv(file)
    sc2df.dropna(axis=0, inplace=True)
    index = sc2df[sc2df["TotalHours"] == sc2df[
        "TotalHours"].max()].index.tolist()
    sc2df.drop(index, inplace=True)
    #sc2df['LeagueIndex'] = sc2df['LeagueIndex'].astype('category')\
    #    .cat.rename_categories([str(level) for level in sc2df['LeagueIndex'].cat.categories])
    sc2df['LeagueIndex'] = sc2df['LeagueIndex'].astype('str')
    sc2df.drop(['GameID'], axis=1, inplace=True)
                # 'NumberOfPACs',
                # 'GapBetweenPACs',
                # 'ActionLatency',
                # 'ActionsInPAC'], axis=1, inplace=True)
    return sc2df
