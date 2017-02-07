import pandas as pd


def mixed(path='data/zensus/zensus.csv'):
    df = pd.read_csv(path, delimiter=';')
    #print(df.head())
    df['Bundesland'] = df['Bundesland'].astype('category')
    df['Landkreis'] = df['Landkreis'].astype('category')
    return df

if __name__ == '__main__':
    mixed('zensus.csv')