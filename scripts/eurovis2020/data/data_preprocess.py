import pandas as pd

if __name__ == '__main__':
    titanic_path_orig = './titanic_orig.csv'
    titanic_path_preprocessed = './titanic.csv'

    titanic = pd.read_csv(titanic_path_orig, index_col=None)
    titanic.pclass.replace({1:'first', 2:'second', 3:'third'}, inplace=True)
    titanic.survived.replace({1:'yes', 0:'no'}, inplace=True)
    titanic.has_cabin_number.replace({1:'yes', 0:'no'}, inplace=True)
    del titanic['sibsp']
    del titanic['parch']
    titanic.to_csv(titanic_path_preprocessed, index=None)
