import pandas as pd
from mb_modelbase.utils import data_import_utils

df = pd.read_csv('allbus_cleaned.csv', index_col=None)

# make spectrum categorical by converting all to string
data_import_utils.to_string_cols(df, ['spectrum'], inplace=True)

# make age categorical by binning it into 10 equi-sized bin and converting them to accordingly named strings
df.age = data_import_utils.to_binned_stringed_series(df.age, 10)

# now learn a model with this data
from mb_modelbase.models_core.mixable_cond_gaussian import MixableCondGaussianModel
from mb_modelbase.models_core.empirical_model import EmpiricalModel

emp_model = EmpiricalModel('allbus_test_emp').fit(df=df)

model = MixableCondGaussianModel("allbus_test_mcg").fit(df=df, empirical_model_name='allbus_test_emp')

print(str(model.names))
for f in model.fields:
    print('{} with domain {}'.format(f['name'], str(f['extent'])))
print(model.aggregate(method='maximum'))

emp_model.save('.')
model.save('.')

## note that pandas tends to parse strings into number when reading from a csv, regardless of quotation...
# # csv.QUOTE_NONNUMERIC will assure quote marks around those strings that are numbers
# df.to_csv('./foobar.csv', index=None, quoting=csv.QUOTE_NONNUMERIC)
# df2 = pd.read_csv('./foobar.csv', index_col=None)
# df3 = pd.read_csv('./foobar.csv', index_col=None, quoting=csv.QUOTE_NONE, header=0)
# print(df2.head())
# print(df3.head())
# pass
