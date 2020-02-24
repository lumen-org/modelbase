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

    def preStart(self):
        pass

    def postRun(self):
        pass

    def run(self, instance) -> float:
        self.preStart()
        start = time.process_time()
        self._run(instance=instance)
        end = time.process_time()
        t = end - start
        if self.n > 1:
            t /= self.n
        self.postRun()
        return t
