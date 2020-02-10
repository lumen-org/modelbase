from mb_modelbase.server.modelbase import ModelBase
from mb_modelbase.utils.benchmark import ModelbaseActivityfileBenchmark
from mb_modelbase.utils.benchmark import Benchmark
import numpy as np
import pandas as pd
import mb_modelbase.cache.cache as mc
import time
import json
import abc
import plotly.express as px



# class SimpleBenchmark(Benchmark):
#     def __init__(self):
#         Benchmark.__init__(self, 'Simple Benchmark')
#         self.queries = [
#             {'FROM': 'mcg_allbus_map',
#              'SPLIT BY': [{'name': 'sepal_length', 'split': 'equiinterval', 'args': [25], 'class': 'Split'}],
#              'PREDICT': ['sepal_length', {'name': ['sepal_length'], 'aggregation': 'probability', 'class': 'Density'}]}
#         ]
#
#     def _run(self, instance: ModelBase) -> float:
#         [instance.execute(q) for q in self.queries]


if __name__ == '__main__':
    model_dir = "/home/leng_ch/git/lumen/fitted_models"
    no_cache = ModelBase(
        name='no_cache',
        model_dir=model_dir,
        cache=None)

    dict_cache = ModelBase(
        name='dict_cache',
        model_dir=model_dir,
        cache=mc.DictCache())

    # mem_cache = mbase.ModelBase(
    #     name='no_cache',
    #     model_dir='../../../fitted_models',
    #     cache=mc.MemcachedCache())

    bases = [
        no_cache,
        dict_cache
    ]

    benchmarks = [
        ModelbaseActivityfileBenchmark
    ]

    results = pd.DataFrame({
        base.name: [benchmark().run(base) for benchmark in benchmarks] for base in bases
    })







