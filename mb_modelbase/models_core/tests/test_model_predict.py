
from mb_modelbase.models_core import Model
from mb_modelbase.models_core.base import *
from mb_modelbase.models_core.mixable_cond_gaussian import MixableCondGaussianModel as MixCondGauss

from mb_modelbase.models_core.tests import test_crabs as crabs

if __name__ == '__main__':
    model = Model.load('crabs_test.mdl')

    # data = crabs.mixed()
    # model = MixCondGauss("TestMod").fit(df=data)
    # Model.save(model, 'crabs_test.mdl')

    print(str(model.names))
    # prints: ['species', 'sex', 'FL', 'RW', 'CL', 'CW', 'BD']

    sex = model.byname('sex')
    species = model.byname('species')

    res = model.predict(['sex', 'species', Density([sex, species])], splitby=[Split(sex), Split(species)])

    print(res)
