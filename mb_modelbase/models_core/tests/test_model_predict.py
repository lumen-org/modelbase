
from mb_modelbase.models_core import Model
from mb_modelbase.models_core.base import *
from mb_modelbase.models_core.mixable_cond_gaussian import MixableCondGaussianModel as MixCondGauss

from mb_modelbase.models_core.tests import test_crabs as crabs

if __name__ == '__main__':
    model = Model.load('crabs_test.mdl')
    model.parallel_processing = False

    # data = crabs.mixed()
    # model = MixCondGauss("TestMod").fit(df=data)
    # Model.save(model, 'crabs_test.mdl')

    print(str(model.names))
    # prints: ['species', 'sex', 'FL', 'RW', 'CL', 'CW', 'BD']

    sex = model.byname('sex')
    species = model.byname('species')
    FL = model.byname('FL')
    RW = model.byname('RW')

    res = model.predict(['sex', 'species', Density(sex), Density(species)], splitby=[Split(sex), Split(species)])
    print(res)

    res = model.predict(['species', Density(species)], splitby=Split(species))
    print(res)

    res = model.predict(['sex', 'species', Density([sex, species])], splitby=[Split(sex), Split(species)])
    print(res)

    res = model.predict(['sex', 'RW', Aggregation(FL, method='maximum', yields='FL')], splitby=[Split(sex), Split(RW, method='equidist')])
    print(res)

    res = model.predict(['species', Density(species)], splitby=Split(species))
    print(res)

    res = model.predict(['FL', Density(FL)], splitby=Split(FL, method='equidist'))
    print(res)

    res = model.predict([Aggregation(FL, method='maximum', yields='FL')])
    print(res)

    res = model.predict([Aggregation(FL, method='maximum', yields='FL')], splitby=[Split(sex)])
    print(res)

    res = model.predict(['sex', Aggregation(FL, method='maximum', yields='FL')], splitby=[Split(sex)])
    print(res)

    res = model.predict(['FL', Probability(FL)], splitby=Split(FL, method='equiinterval'))
    print(res)

    res = model.predict(['sex', Density(sex)], splitby=Split(sex))
    print(res)

    res = model.predict(['sex', Density(sex), Aggregation(FL, method='maximum', yields='FL')], splitby=[Split(sex)])
    print(res)

