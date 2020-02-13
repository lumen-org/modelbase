from mb_modelbase.utils.benchmark.Benchmark import Benchmark
from mb_modelbase.server.modelbase import ModelBase
import json

class ModelbaseActivityfileBenchmark(Benchmark):
    def __init__(self, file='benchmarkInteraction2.log'):
        Benchmark.__init__(self,name=file)
        with open(file,'r') as f:
            lines = list(f)

        self.queries = [
            json.loads(x[:-1])
            for x in lines
        ]

    def _run(self, instance: ModelBase) -> float:
        [instance.execute(q) for q in self.queries]


