from mb_modelbase.server import modelbase as mbase
import numpy as np
import pandas as pd
import mb_modelbase.cache.cache as mc
import time
import abc


class Benchmark(abc.ABC):
    def __init__(self,name):
        self.name = name

    @abc.abstractmethod
    def _run(self, modelbase):
        """
            takes a modelbase and runs the benchmark
            @param modelbase a modelbase object
        """
    def run(self,modelbase):
        start = time.time()
        self._run(modelbase=modelbase)
        end = time.time()
        return end - start

class SimpleBenchmark(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, 'Simple Benchmark')

    def _run(self, modelbase):
        queries = [
            {'FROM': '__mcg_iris_map_0_0',
             'SPLIT BY': [{'name': 'sepal_length', 'split': 'equiinterval', 'args': [25], 'class': 'Split'}],
             'PREDICT': ['sepal_length', {'name': ['sepal_length'], 'aggregation': 'probability', 'class': 'Density'}]}

        ]
        [modelbase.execute(q) for q in queries]



if __name__ == '__name__':
    no_cache = mbase.ModelBase(
        name='no_cache',
        model_dir='../../../fitted_models',
        cache=None)

    dict_cache = mbase.ModelBase(
        name='no_cache',
        model_dir='../../../fitted_models',
        cache=mc.DictCache())

    mem_cache = mbase.ModelBase(
        name='no_cache',
        model_dir='../../../fitted_models',
        cache=mc.MemcachedCache())

    bases = [
        no_cache,
        dict_cache,
        mem_cache
    ]

    benchmarks = [
        SimpleBenchmark()
    ]

    for ba in bases:
        for be in benchmarks
