# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import pandas as pd
#from ..cleansing import printDataAndParameterQuantity
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".full.cleansed"


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
            print(df[col].name, " (", df[col].dtype,
                  "), values counts: ", df[col].value_counts())

    return df


# data preperation of the adults data set for use in a purely categorical model
def categorical(datafilepath=_csvfilepath, verbose=False):

    df = _loadfromfile(datafilepath, verbose)

    # create dataframe with only the categorical columns
    categorial_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                          'native-country']
    dfcat = pd.DataFrame(df, columns=categorial_columns)

    # print histogram for each category
    if verbose:
        import matplotlib.pyplot as plt
        for col in dfcat.columns:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            dfcat[col].value_counts().plot.bar(ax=ax1)

    # after inspection of these, do the following additional transformations

    # column 'workclass' seems useful
    # column 'education' seems useful, but has many possible values
    # -> summarize into 'low', 'middle' and 'high'

    # todo: easier & cleaner: adaption of the following
    # adult_set.workclass.replace(['Without-pay','Never-worked'], ['unemp']*2, inplace = True)
    edu = dfcat.education
    low = ('Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th')
    middle = ('11th', '12th', 'HS-grad', 'Some-college',
              'Prof-school', 'Assoc-voc')
    high = ('Assoc-acdm', 'Bachelors', 'Masters', 'Doctorate')
    edu.loc[[x in low for x in edu]] = 'Low'
    edu.loc[[x in middle for x in edu]] = 'Middle'
    edu.loc[[x in high for x in edu]] = 'High'

    # column 'marital status' seems a little useless somehow
    # -> remove it!
    del dfcat['marital-status']

    # column 'occupation' has many possible values but seems useful for now
    # ... delete it, we get too many parameters to estimate
    del dfcat['occupation']

    # column 'relationship' seems useful

    # column 'race' is 'white' or 'black' for the vast majority
    # -> summarize into 3 categories: white, black, other
    race = dfcat.race  # this is NOT a copy but a reference to the column!
    mask = (race != "White") & (race != "Black")
    race.loc[mask] = 'Other'

    # column 'sex' seems useful

    # column 'native-country' is 'United-States' for the vast majority
    # -> replace all but United States country by 'Outside-US'
    country = dfcat['native-country']
    country[country != 'United-States'] = 'Outside-US'

    # actually turn them into categorical columns
    for col in dfcat.columns:
        dfcat[col] = dfcat[col].astype('category')

    # number of parameters to estimate
#    if verbose:
#        printDataAndParameterQuantity(dfcat)

    # TODO: discretize any continuous columns to categorical?
    return dfcat


def cat4_cont5(datafilepath=_csvfilepath, verbose=False):

    # File: adult.full.cleansed
    df = pd.read_csv(datafilepath, index_col=False, skipinitialspace=True)

    del df['relationship']
    del df['occupation']
    del df['workclass']
    del df['education']
    del df['fnlwgt']  # What's that?!

    df = df.rename(columns={'native-country': "native_country", "marital-status": "marital_status",
                            "capital-gain": "capital_gain", "capital-loss": "capital-loss", "hours-per-week": "hours_per_week"})

    # Replace all values which are NOT "United-States" with "Outside-US" in column native_country
    df.replace(to_replace={"native_country": {
               "^((?!United).)*$": "Outside-US"}}, inplace=True, regex=True)
    # Reduce column "marital_status" to three categoricals
    df.replace(to_replace={"marital_status": {"Married.*": "Married",
                                              "Separated": "Divorced", "Widowed": "Divorced"}}, inplace=True, regex=True)
    # Reduce column "race" to three categoricals
    df.race.replace(to_replace={
                    "Amer-Indian-Eskimo": "Other", "Asian-Pac-Islander": "Other"}, inplace=True)

    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    return df
