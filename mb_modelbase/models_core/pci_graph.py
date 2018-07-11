# Copyright (c) 2018 Philipp Lucas (philipp.lucas@uni-jena.de)
"""
@author: Philipp Lucas
"""
#from mb_modelbase.models_core import Model
from CGmodelselection.graph import get_graph_from_data

from mb_modelbase.utils import update_opts


def create(df, **kwargs):
    """Create and return a pair-wise conditionally independence graph for given DataFrame df."""

    valid_create_opts = {
        'standardize': [True, False],
        'model': ['PW', 'CLZ'],
        'disp': [False]
    }
    default_create_opts = {
        'standardize': True,  # standardize data before learning (recommended)
        'model': 'PW',   # choose from 'PW' (pairwise model) and 'CLZ' (CLZ model with triple interactions)
        'graphthreshold': 1e-1,  # trade-off parameter for l1-regularization term
        'kS': 2,  # regularization parameter for l1 regularization
        'disp': False,
    }
    opts = update_opts(default_create_opts, kwargs, valid_create_opts)

    grpnormmat, graph, dlegend = get_graph_from_data(df, **opts)
    return {
        "weight_matrix": grpnormmat,
        "binary_matrix": graph,
        "dimension_label": dlegend,
        "opts": opts,
    }


def to_json(pci_graph):
    """Convert and return pci_graph into a serializable dict that has keys with values as follows:
        'nodes': list of string labels of nodes
        'edges': list of dict with keys source, target, weight
    """
    weights = pci_graph['weight_matrix']
    binary = pci_graph['binary_matrix']
    labels = pci_graph['dimension_label']
    edges = []

    for i in range(weights.shape[0]):
        for j in range(i, weights.shape[1]):
            if binary[i,j]:
                edges.append({
                    'source': labels[i],
                    'target': labels[j],
                    'weight': weights[i, j],
                })
    return {
        'nodes': labels,
        'edges': edges,

    }