import numpy as np

def as_numeric(df):
    value_dict = {}
    df_new = df.copy()
    for column in df:
        unique_values = np.unique(np.array(df[column]))
        if any([isinstance(i, str) for i in unique_values]):
            value_to_number = {value: number for number, value in enumerate(unique_values)}
            value_dict[column] = value_to_number
            df_new[column] = df[column].replace(value_to_number)
    return value_dict, df_new