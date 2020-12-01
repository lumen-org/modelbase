# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".cleaned.cleveland.csv"

"""Mapping of columns to data type, induced from heart-disease.names:
0,f      -- 1. #3  (age)
1,c      -- 2. #4  (sex)
2,c      -- 3. #9  (cp): chest_pain
3,f      -- 4. #10 (trestbps): blood_pres
4,f      -- 5. #12 (chol)
5,c      -- 6. #16 (fbs): blood_sugar
6,c      -- 7. #19 (restecg)
7,f      -- 8. #32 (thalach): max_heartrate
8,c      -- 9. #38 (exang): angina
9,f      -- 10. #40 (oldpeak)
10,c      -- 11. #41 (slope)
11,i      -- 12. #44 (ca): vessels_cnt
12,c      -- 13. #51 (thal)
13,c      -- 14. #58 (num): disease
f ... float (continuous), c ... categorical, i ... integer (ordered)
"""


# handy shortcut: close all open figures
#plt.close("all")


def _loadfromfile(datafilepath=_csvfilepath, verbose=False):
    # load
    df = pd.read_csv(datafilepath, index_col=False, skipinitialspace=True)

    # drop NA/NaN
    # datafilepath = 'data/adult/adult.full'
    # df = pd.read_csv(datafilepath, index_col=False, na_values='?')
    # dfclean = df.dropna()

    # print information about columns:
    if verbose:
        print("Columns and their data type:")
        for col in df.columns:
            print(df[col].name, " (", df[col].dtype, "), values counts: ", df[col].value_counts())

    return df


# data preparation of the heart disease data set for use in a purely categorical model
def categorical(datafilepath=_csvfilepath, verbose=False):
    df = _loadfromfile(datafilepath, verbose)

    catidx = [1, 2, 5, 6, 8, 10, 12, 13]
    intidx = [11]
    contidx = [0, 3, 4, 7, 9]

    # create dataframe with only the continuous columns
    dfcont = df.iloc[:, contidx].copy()
    if verbose:
        dfcont.hist()

    # create dataframe with only the categorical columns
    dfcat = df.iloc[:, catidx].copy()
    if verbose:
        dfcat.head()

    # turn it in useful categories
    convert = {
        "sex": {1.0: "male", 0.0: "female"},
        "chest_pain": {1.0: "typical angina", 2.0: "atypical angina", 3.0: "non-anginal pain", 4.0: "asymptomatic"},
        "blood_sugar": {0.0: "low", 1.0: "high"},
        "restecg": {0.0: "normal", 1.0: " ST-T wave abnormality", 2.0: "ventricular hypertrophy"},
        "angina": {0.0: "False", 1.0: "True"},
        "slope": {1.0: "upsloping", 2.0: "flat", 3.0: "down"},
        "thal": {3.0: "normal", 6.0: "fixed defect", 7.0: "reversable defect"},
        "disease": {0: "False", 1: "True", 2: "True", 3: "True", 4: "True"},
    }

    for colname in convert:
        levelmap = convert[colname]
        for key in levelmap:
            dfcat.loc[dfcat[colname] == key, colname] = levelmap[key]
    if verbose:
        dfcat.head()

    # print histogram for each category
    if verbose:
        import matplotlib.pyplot as plt
        for col in dfcat.columns:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            dfcat[col].value_counts().plot.bar(ax=ax1)

    # actually turn them into categorical columns
    for col in dfcat.columns:
        dfcat[col] = dfcat[col].astype('category')

    # TODO: discretize any continuous columns to categorical?

    return dfcat