# cython: language_level = 3
# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
# cython: profile = False

from collections import OrderedDict
import numpy as np
import pandas as pd
import copy
from anytree import Node, RenderTree, search
cimport numpy as np

rng = None

cdef int id_counter = 0
cdef dict id_mapping = {}
cdef dict reverse_id_mapping = {}

cdef int get_id(tuple item):
    """"Each unique parameter tuple (mu, sigma) or (choice1, choice2, ..., choice_n) gets an ID (sort of a hash).
    This let's numpy determine unique elements much faster
    """

    global id_counter
    global id_mapping
    global reverse_id_mapping

    if item in id_mapping:
        return id_mapping[item]
    else:
        id_counter += 1
        id_mapping[item] = id_counter
        reverse_id_mapping[id_counter] = item
        return id_counter


cdef list roll(tuple bias_list, int n):
    """make a random, weighted choice

    Example: Input [0.1, 0.9] will return most of the time 1 and sometimes 0 (the index of the input array).

    Parameters:
        bias_list (tuple[double]): The probabilites from where we choose.
        n (int): How often do we choose

    Returns:
        list[int]: The random choices
    """
    cdef Py_ssize_t length = len(bias_list)
    # convert to numpy array
    cdef np.ndarray[double, ndim=1] probs = np.array(bias_list)
    # normalize to a sum of 1, because the source data is not always accurate and exactly 1, which causes numpy to complain
    probs /= probs.sum()
    return list(rng.choice(length, n, p=probs))


def generate_samples(trees, conditional_nodes, n, random):
    global rng
    # use the individually seeden rng
    rng = random
    length = len(conditional_nodes)
    number_name_mapping = [0] * length
    name_number_mapping = {}
    for i,conditional_node in enumerate(conditional_nodes):
        number_name_mapping[i] = conditional_node
        name_number_mapping[conditional_node] = i

    data = _generate_samples(trees, number_name_mapping, name_number_mapping, length, n)
    return data


def get_key(row, key, current_node, number_name_mapping, name_number_mapping):
    cdef bint is_bias_roll = False
    cdef Py_ssize_t child_count
    cdef str parent_type
    cdef double parent_normal_dist_result
    cdef double bias
    cdef double sigma
    cdef int index_decision
    cdef int leading_decision
    cdef list bias_list

    while True: # loop gets finished with return
        node_children = current_node.children
        child_count = len(node_children)
        if (child_count == 0): # node should be -> current_node.normal
            # use factor of parent to calulate normal distribution
            if (current_node.parent.factor):
                parent_type = current_node.parent.name
                parent_normal_dist_result = row[name_number_mapping[parent_type]]
                bias = parent_normal_dist_result * current_node.parent.factor
            else:
                bias = 0
            mu = bias + current_node.parameter[0]
            sigma = current_node.parameter[1]
            return False, get_id((mu, sigma))

        elif child_count == 1: # when one child: walk down tree
            current_node = node_children[0]

        else: # multiple children
            if node_children[0].parameter is None: # können wir davon ausgehen, dass alle children von gleicher art sind??
                index_decision = name_number_mapping[node_children[0].name] # get the index of the node next in tree
                leading_decision = row[index_decision] # previous made decision now used to advance in tree
                found_node = None
                for child in node_children:
                    if child.case == leading_decision:
                        found_node = child
                        break
                if found_node == None:
                    print("invalid tree")
                    exit()
                current_node = found_node

            else:
                bias_list = []
                for child in node_children:
                    bias_list.append(child.parameter)
                return True, get_id(tuple(bias_list))


def _generate_samples(trees, number_name_mapping, name_number_mapping, length, n):
    # fix OS datatype mismatch: https://stackoverflow.com/questions/32262976/cython-buffer-type-mismatch-expected-int-but-got-long
    # https://docs.scipy.org/doc/numpy/user/basics.types.html
    cdef np.ndarray[long long, ndim=2] data = np.empty((n, length), dtype = np.longlong)
    cdef np.ndarray result = np.empty((n, length), dtype = object)
    cdef int col_index
    cdef bint is_bias_roll
    cdef str key
    cdef int row_index
    cdef np.ndarray column
    cdef np.ndarray[long long, ndim=1] uniques
    cdef np.ndarray[long long, ndim=1] reverse_indices
    cdef np.ndarray[long long, ndim=1] counts
    # not a numpy array because we want to keep the .pop() function
    cdef list randoms
    cdef int i_uniques
    cdef np.int count
    cdef double mu
    cdef double sigma
    cdef tuple parameter

    for col_index in range(length): # loop over each of the columns
        is_bias_roll = False
        key = number_name_mapping[col_index]
        for row_index in range(n): # loop over the row of a column
            current_node = trees[key]
            is_bias_roll, data[row_index][col_index] = get_key(data[row_index], key, current_node, number_name_mapping, name_number_mapping)

        column = data[:,col_index]
        uniques, reverse_indices, counts = np.unique(column, return_inverse=True, return_counts=True)
        randoms = [0] * len(uniques)
        for i_uniques, unique in enumerate(uniques):
            count = counts[i_uniques]
            parameter = reverse_id_mapping[uniques[i_uniques]] # get the parameter tuple back from the ID
            if is_bias_roll: # choice
                randoms[i_uniques] = roll(parameter, count)
            else: # normal distribution
                mu = parameter[0]
                sigma = parameter[1]
                randoms[i_uniques] = list(rng.normal(mu, sigma, count))

        for i_reverse_index, reverse_index in enumerate(reverse_indices):
            column[i_reverse_index] = randoms[reverse_index].pop()
        result[:,col_index] = column
    return result
