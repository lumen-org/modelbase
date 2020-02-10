from mb_modelbase.utils.benchmark.Benchmark import Benchmark
from mb_modelbase.server.modelbase import ModelBase
import json

class ModelbaseActivityfileBenchmark(Benchmark):
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

    def _run(self, instance: ModelBase) -> float:
        [instance.execute(q) for q in self.queries]


