import abc
import time


class Benchmark(abc.ABC):
    """Abstract baseclass for a Benchmark

    Override _run and implement the logic that is to be timed.
    If the number of runs is larger than one,
    then the time returned by the method run is the average time of n runs.

    Attributes:
        _name (str): name of the benchmark
        _n (int): number of runs to be performed and averaged
    Args:
        name (str): name of the benchmark
        n (int): number of runs to be performed and averaged
    """

    def __init__(self, name, n=1):
        self._name = name
        self._n = n

    @abc.abstractmethod
    def _run(self, instance):
        """This method takes an object and performs the actions to be benchmarked.

            Args:
                instance: An object to be benchmarked
        """

    def preRun(self):
        """Setup benchmark environment if necessary"""
        pass

    def postRun(self):
        """Shutdown benchmark environment if necessary"""
        pass

    def run(self, instance) -> float:
        """This method runs the benchmark.

        Args:
            instance: An object to be benchmarked.
        Returns:
            float: The time the execution of the benchmark took.
        """
        self.preRun()
        start = time.process_time()
        self._run(instance=instance)
        end = time.process_time()
        t = end - start
        if self._n > 1:
            t /= self._n
        self.postRun()
        return t
