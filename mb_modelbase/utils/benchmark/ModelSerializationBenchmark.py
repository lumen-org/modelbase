from mb_modelbase.utils.benchmark.Benchmark import Benchmark
from mb_modelbase.core.models import Model


class ModelSerializationBenchmark(Benchmark):
    """A Benchmark to measure the overhead of serialization in reading and writing.

    Implements abstract class Benchmark.

    Attributes:
        _cache (BaseCache): The cache to be benchmarked.

    Args:
        cache (BaseCache): The cache to be benchmarked.
        n (int): number of runs to be performed and averaged.

    Todo:
        * Store model at randomized keys to simulate random access
    """

    def __init__(self, cache, n=1000):
        Benchmark.__init__(self, 'ModelSerializationBenchmark', n=n)
        self._cache = cache

    def _run(self, instance: Model):
        """ perform  """
        for i in range(self._n):
            self._cache.set('test', instance)
            res = self._cache.get('test')
