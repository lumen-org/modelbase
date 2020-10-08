# import mb_modelbase as mb
# from mb_modelbase.core.mixable_cond_gaussian import MixableCondGaussianModel
# from mb_modelbase.core.cond_gaussian_wm import CgWmModel as CGWModel
# from mb_modelbase.core.tests import crabs
#
#
# def _test_density(model, split_cnt=200, eps=0.05):
#     # Calculates the accumulated 1d density along this 1-dimensional model and succeeds iff this adds up to 1 +- eps.
#     if model.dim != 1:
#         #raise NotImplementedError('Currently this can only test density for 1d models/data')
#         return
#
#     mvd = model._modeldata_field
#     f = model.fields[0]
#
#     def check_for_mode(mode):
#         p = model.predict(predict=[mb.Density(f)],
#                           splitby=[mb.Split(f, args=split_cnt)],
#                           where=[mb.Condition(mvd, "==", mode)])
#         p_sum = p.sum().values
#         if abs(p_sum - 1) > eps:
#             raise AssertionError(str(mode) + " prob does not add up to 1. It is " + str(p_sum))
#
#     if model.mode == 'both' or model.mode == 'data':
#         check_for_mode('data')
#     if model.mode == 'both' or model.mode == 'model':
#         check_for_mode('model')
#
#
# if __name__ == '__main__':
#
#     df = crabs.continuous()[['RW']]
#     # model = MixableCondGaussianModel("foo").fit(df)
#     model = CGWModel("foo").fit(df)
#     _test_density(model)