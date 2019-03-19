# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas

Automatically determines suitable extents of dimensions such that the
extent covers the non-zero interval of the marginal model for that
dimension.


"""

from mb_modelbase.models_core import domains as dm


def print_extents(model):
    for field in model.fields:
        print(field['name'])
        print(field['extent'])


def field_to_auto_extent(model, dim_name, prec=1e-03, step=.02):
    """Extends the extent of the quantitative dimension with name dim_name of model such that the range covers all non-zero values of the marginal model on dim_name.
    It returns the extended range for that dimension as a 2-element tuple (low, high)

    Args:
        model: A Model
        dim_name: name of a quantitative dimension of model
        [prec]: Precision
        [step]: step size as a factor of the extent of the dimension.
    """

    # check if dim_name is in model
    field = model.byname(dim_name)
    print("--------" + field['name'] + "--------")
    if field['dtype'] == 'string':
        raise ValueError("dimension may not be categorical")

    # compute marginal model
    mm = model.marginalize(keep=dim_name)

    # get current extent
    extent = mm.byname(dim_name)['extent'].values()
    # go in steps of step-%
    len_step = (extent[1] - extent[0])*step

    low = extent[0]
    while mm.density([low]) > prec:
        low -= len_step

    high = extent[1]
    while mm.density([high]) > prec:
        high += len_step

    return low, high


def auto_range_by_sample(**kwargs):
    """Idea: sample from the model very often and just take the resulting extent"""
    raise NotImplementedError


def adopt_all_extents(model, how=field_to_auto_extent):

    for field in model.fields:
        if field['dtype'] != 'string':
            extent = how(model.copy(), field['name'])
            field['extent'] = dm.NumericDomain(extent)  # modifies model!
    return model