from mb_modelbase.server import modelbase as mbase
import numpy as np
import pandas as pd
import mb_modelbase.cache.cache as mc
import time
import json
import abc
import plotly.express as px

class Benchmark(abc.ABC):
    def __init__(self,name):
        self.name = name

    @abc.abstractmethod
    def _run(self, modelbase: mbase.ModelBase):
        """
            takes a modelbase and runs the benchmark
            @param modelbase a modelbase object
        """
    def run(self, modelbase: mbase.ModelBase) -> float:
        start = time.time()
        self._run(modelbase=modelbase)
        end = time.time()
        return end - start

class SimpleBenchmark(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, 'Simple Benchmark')
        self.queries = [
            {'FROM': 'mcg_allbus_map',
             'SPLIT BY': [{'name': 'sepal_length', 'split': 'equiinterval', 'args': [25], 'class': 'Split'}],
             'PREDICT': ['sepal_length', {'name': ['sepal_length'], 'aggregation': 'probability', 'class': 'Density'}]}
        ]

    def _run(self, modelbase):
        [modelbase.execute(q) for q in self.queries]

class ActivityFileBenchmark(Benchmark):
    def __init__(self, file='benchmarkInteraction2.log'):
        Benchmark.__init__(self,name=file)
        with open(file,'r') as f:
            lines = list(f)

        noNewLine = list(map(lambda x: x[:-1], lines))
        logsOnly = list(filter(lambda x: x != 'CACHE HIT!' and x != 'CACHE MISS!' and len(x.split('QUERY:')) == 2 , noNewLine))
        msgOnly = list(map(lambda x: x.split('QUERY:')[1], logsOnly))
        doubleQuotes = list(map(lambda x: x.replace("'", '"'), msgOnly))
        jsons = list(map(json.loads, doubleQuotes))
        self.queries = list(filter(lambda x: 'SHOW' not in x.keys(), jsons))
        #self.queries = [json.loads(line) for line in lines]
        #self.queries = self.queries + self.queries

    def _run(self, modelbase):
        [modelbase.execute(q) for q in self.queries]

if __name__ == '__main__':
    no_cache = mbase.ModelBase(
        name='no_cache',
        model_dir='../../../fitted_models',
        cache=None)

    dict_cache = mbase.ModelBase(
        name='dict_cache',
        model_dir='../../../fitted_models',
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
        ActivityFileBenchmark
    ]

    results = pd.DataFrame({
        base.name: [benchmark().run(base) for benchmark in benchmarks] for base in bases
    })







