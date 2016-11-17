import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

# handy shortcut: close all open figures
#plt.close("all")

# data preperation of the adults data set for use in a purely categorical model

# open data file
datafilepath = 'data/bank/bank-full.csv'
df = pd.read_csv(datafilepath, delimiter=";", index_col=False, skipinitialspace=True)

# drop NA/NaN
#datafilepath = 'data/adult/adult.full'
#df = pd.read_csv(datafilepath, index_col=False, na_values='?')
#dfclean = df.dropna()

# print information about columns:
print("Columns and their data type:")
for col in df.columns:
    #print(df[col].name, " (", df[col].dtype, "), values counts: ", df[col].value_counts())
    print(df[col].name, " (", df[col].dtype)

# print histograms on continuous columns
df.hist()