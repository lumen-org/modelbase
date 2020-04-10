import abc
import time


class Benchmark(abc.ABC):
    """Abstract baseclass for a Benchmark

    Override _run and implement the logic that is to be timed

    Attributes:
        _name (str): name of the benchmark
        _n (int): number of runs to be performed and averaged
    """

    def __init__(self, name, n=1):
        self._name = name
        self._n = n

    @abc.abstractmethod
    def _run(self, instance):
        """takes an instance and runs the benchmark
            @param instance an object
        """

    def preRun(self):
        """Setup benchmark environment if necessary"""
        pass

    def postRun(self):
        """Shutdown benchmark environment"""
        pass

    def run(self, instance) -> float:
        self.preRun()
        start = time.process_time()
        self._run(instance=instance)
        end = time.process_time()
        t = end - start
        if self._n > 1:
            t /= self._n
        self.postRun()
        return t
