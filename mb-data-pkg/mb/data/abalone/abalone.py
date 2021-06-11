import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"

try:
    import spn.structure.leaves.parametric.Parametric as spn_parameter_types
    import spn.structure.StatisticalTypes as spn_statistical_types
except ImportError:
    pass


def cg(file=_csvfilepath):
    df = pd.read_csv(file)
    df.dropna(axis=0, inplace=True)
    return df

def spn_parameters():
    abalone_variable_types = {
        'Sex': spn_parameter_types.Categorical,
        'Length': spn_parameter_types.Gaussian,
        'Diameter': spn_parameter_types.Gaussian, 'Height': spn_parameter_types.Gaussian,
        'WholeWeight': spn_parameter_types.Gaussian, 'ShuckedWeight': spn_parameter_types.Gaussian,
        'VisceraWeight': spn_parameter_types.Gaussian, 'ShellWeight': spn_parameter_types.Gaussian,
        'Rings': spn_parameter_types.Categorical
    }
    return abalone_variable_types

def spn_metatypes():
    abalone_variable_types = {
        'Sex': spn_statistical_types.MetaType.DISCRETE,
        'Length': spn_statistical_types.MetaType.REAL,
        'Diameter': spn_statistical_types.MetaType.REAL, 'Height': spn_statistical_types.MetaType.REAL,
        'WholeWeight': spn_statistical_types.MetaType.REAL, 'ShuckedWeight': spn_statistical_types.MetaType.REAL,
        'VisceraWeight': spn_statistical_types.MetaType.REAL, 'ShellWeight': spn_statistical_types.MetaType.REAL,
        'Rings': spn_statistical_types.MetaType.DISCRETE
    }
    return abalone_variable_types

def mixed(filepath=_csvfilepath):
    """Loads the abalone data set from a csv file, removes the index column and returns the
    remaining data as a pandas data frame
    """
    df = pd.read_csv(filepath)
    df.dropna(axis=0, inplace=True)
    #df = df.replace({'M': 0, 'F': 1, 'I': 2})
    df['Sex'] = df['Sex'].astype('category')
    df['Rings'] = df['Rings'].astype('category')
    return df


if __name__ == "__main__":
    types = spn_statistical_types
    print(types)


