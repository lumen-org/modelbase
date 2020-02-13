from mb_modelbase.utils.benchmark.Benchmark import Benchmark
from mb_modelbase.models_core.models import Model

class ModelSerializationBenchmark(Benchmark):
    def __init__(self,cache,n=100):
        Benchmark.__init__(self, 'ModelSerializationBechmark',n=n)
        self.cache = cache

    def _run(self, instance: Model):
        for i in range(self.n):
            self.cache.set('test',instance)
            res = self.cache.get('test')



