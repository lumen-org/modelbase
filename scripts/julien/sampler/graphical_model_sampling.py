#! /usr/bin/python3.7
# -*- coding: utf-8 -*-

"""This program is part of the project required for the lecture Algorithm Engineering Lab of the Friedrich-Schiller-University of Jena.

As input for this program a JSON file is necessary. This JSON file holds information about multiple tree structures.
Each of those trees are linked together and have to be processed in a given order. The result for each tree is a number
identifying the node in the bottom of the tree. In order to get the number of the node, the each tree has to be traversed
in the correct order from top to bottom. Each subnode of a tree is chosen based on a given probability for each subnode.
To get a bottom node number of a tree later in the order, it's also necessary to know, which number got rolled from the
other trees.

The result of the program execution is a CSV file containing a so called sample, which holds the resulting number of
each tree. Each sample is unique (with a high probability).

Please use the following command to see all arguments available.
> python graphical_model_sampling.py -h

An example execution could be the following (with output existing directory):
> python graphical_model_sampling.py data/ALLBUS/example.json -s 10 -o output
"""

from pathlib import Path
import argparse
import cProfile
import json
import multiprocessing as mp
import numpy
import os
import pstats
import pandas as pd
import time


from scripts.julien.sampler.create_tree import create_tree
# Cython Modules
from scripts.julien.sampler.gen_samples import generate_samples

def get_arguments(trees, conditional_nodes, sample_count, process_count):
    samples_per_process = sample_count // process_count
    remainder = sample_count % process_count
    arguments = []
    first_run = True
    for _ in range(process_count):
        adjusted_samples_per_process = samples_per_process
        if (first_run):
            adjusted_samples_per_process = samples_per_process + remainder
            first_run = False
        arguments.append((trees, conditional_nodes, adjusted_samples_per_process,
                          numpy.random.RandomState(int.from_bytes(os.urandom(4), byteorder="little"))))
    return arguments


def gen_samples_for_model(n_samples, model_name):
    json_desc_file = Path(f"../julien/learn_pymc3/json_files_for_sampler/{model_name}_sampler.json")
    df = sample(n_samples*10, json_desc_file)

    return df[100:]


def sample(n_samples, json_desc_file):
    with open(json_desc_file, encoding="utf-8-sig") as json_file:
        data = json.load(json_file)
        conditional_nodes = data["conditional node order"]
        trees = {}
        for conditional_node in conditional_nodes:
            trees[conditional_node] = create_tree(data, conditional_node)

    # Generate samples
    process_count = mp.cpu_count()
    start = time.perf_counter()
    # don't use a thread pool when using only one thread
    if (process_count == 1):
        results = generate_samples(*get_arguments(trees, conditional_nodes, n_samples, process_count)[0])
    else:
        results_by_process = mp.Pool(process_count).starmap(generate_samples, get_arguments(
            trees, conditional_nodes, n_samples, process_count))
        results = numpy.concatenate(results_by_process)

    df = pd.DataFrame.from_records(results)
    df.columns = list(conditional_nodes)
    end = time.perf_counter()
    elapsed = end - start

    # more options can be specified also
    print("Generated Samples in", elapsed, "s")
    return df


# Main Guard
if __name__ == "__main__":
    print(sample(1000000, "/home/julien/PycharmProjects/modelbase/scripts/julien/learn_pymc3/json_files_for_sampler/bnlearn_iris_fastiamb_sampler.json"))
