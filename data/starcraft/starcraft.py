import pandas as pd


def cg(file='data/starcraft/starcraft.csv'):
    sc2df = pd.read_csv(file)
    sc2df.dropna(axis=0, inplace=True)
    index = sc2df[sc2df["TotalHours"] == sc2df[
        "TotalHours"].max()].index.tolist()
    sc2df.drop(index, inplace=True)
    #sc2df['LeagueIndex'] = sc2df['LeagueIndex'].astype('category')\
    #    .cat.rename_categories([str(level) for level in sc2df['LeagueIndex'].cat.categories])
    sc2df['LeagueIndex'] = sc2df['LeagueIndex'].astype('str')
    sc2df.drop(['GameID',
                'NumberOfPACs',
                'GapBetweenPACs',
                'ActionLatency',
                'ActionsInPAC'], axis=1, inplace=True)
    return sc2df


if __name__ == "__main__":
    df = cg("starcraft.csv")
