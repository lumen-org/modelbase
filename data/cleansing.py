# Copyright (c) 2016 Philipp Lucas (philipp.lucas@uni-jena.de)
def printDataAndParameterQuantity(df):
    cnt = 1
    for col in df.columns:
        cnt *= len(df[col].value_counts())
    print("number of parameters to estimate: ", str(cnt))
    print("number of observations: ", str(len(df)))
    print("#observations / #params: ", len(df) / cnt)
