import mb_modelbase as mb
import pystan

import pickle
import dill as pickle
import multiprocessing as mp

#
# smb = pystan.StanModel(
#    "/home/me/git/lumen/datasets/mb_data/eight_schools/eight_schools.stan",
# )
#
# with open(
#     "/home/me/git/lumen/datasets/mb_data/eight_schools/eight_schools.stan.pickle", "wb"
# ) as f:
#     pickle.dump(smb, f)
#

sm = pickle.load(
    open(
        "/home/me/git/lumen/datasets/mb_data/eight_schools/eight_schools.stan.pickle",
        "rb",
    )
)
#

evidence = {
    "J": 8,
    "y": [28, 8, -3, 7, -1, 1, 18, 12],
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
}

evidence = {
    "J": 8,
}

mcg_model = mb.Model.load("/home/me/git/lumen/fitted_models/emp_titanic.mdl")
model = mb.StanPPLModel(name="stan_model", stan_model=sm, evidence=evidence)
# model.save("../fitted_models")
# model = mb.Model.load("../fitted_models/stan_model.mdl")

# model = mb.Model.load("/home/me/git/lumen/fitted_models/stan_eight_schools.mdl")

# Set evidence manually


# model.set_data(df=None)

# s = model.stan_model.sampling(data=schools_dat).extract()
# model.fit()
# model._set_data()
ms = model.sample(100)
