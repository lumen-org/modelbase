import abc
import time

class Benchmark(abc.ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def _run(self, instance):
        """
            takes a modelbase and runs the benchmark
            @param instance a modelbase object
        """

    def run(self, instance) -> float:
        start = time.time()
        self._run(instance=instance)
        end = time.time()
        return end - start
