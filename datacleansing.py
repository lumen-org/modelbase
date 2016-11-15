import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# handy shortcut: close all open figures
#plt.close("all")

# data preperation of the adults data set for use in a purely categorical model

# open data file
datafilepath = 'data/adult/adult.full.cleansed'
df = pd.read_csv(datafilepath, index_col=False, skipinitialspace=True)

# drop NA/NaN
#datafilepath = 'data/adult/adult.full'
#df = pd.read_csv(datafilepath, index_col=False, na_values='?')
#dfclean = df.dropna()

# print information about columns:
print("Columns and their data type:")
for col in df.columns:
    print(df[col].name, " (", df[col].dtype, "), values counts: ", df[col].value_counts())

# print histograms on continuous columns
plt.figure()
df.hist()

# age, workclass, fnlwgt, education, education-num, marital-status, occupation,
# relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

# create dataframe with only the categorical columns
categorial_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
dfcat = pd.DataFrame(df, columns=categorial_columns)

# print histogram for each category
for col in dfcat.columns:
    plt.figure()
    dfcat[col].value_counts().plot.bar()

# after inspection of these, do the following additional transformations

# column 'workclass' seems useful
# column 'education' seems useful, but has many possible values
# -> summarize into 'low', 'middle' and 'high'
edu = dfcat.education
low = ('Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th')
middle = ('11th', '12th', 'HS-grad', 'Some-college', 'Prof-school', 'Assoc-voc')
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

# number of paramters to estimate
cnt = 1
for col in dfcat.columns:
    cnt = cnt * len(dfcat[col].value_counts())
print("number of parameters to estimate: ", str(cnt))

# TODO: discretize any continuous columns to categorical?


