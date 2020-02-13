import abc
import time

class Benchmark(abc.ABC):
    def __init__(self, name,n = 1):
        self.name = name
        self.n = n
    @abc.abstractmethod
    def _run(self, instance):
        """
            takes a modelbase and runs the benchmark
            @param instance a modelbase object
        """

    def preStart(self, data=None):
        pass

    def postRun(self, data=None):
        pass

    def run(self, instance) -> float:
        self.preStart()
        start = time.time()
        self._run(instance=instance)
        end = time.time()
        t = end - start
        if self.n > 1:
            t /= self.n
        self.postRun()
        return t
