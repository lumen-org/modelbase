from mb_modelbase.utils.benchmark.Benchmark import Benchmark
import json


class ModelbaseActivityfileBenchmark(Benchmark):
    """Benchmark based on a file with a recorded user interaction.

    This class implements abstract class Benchmark.

    Attributes:
       queries List[DefaultDict] : Queries to be performed
    Args:Params
        file (str): The path to the file containing the benchmark interaction
    """

    def __init__(self, file='benchmarkInteraction2.log'):
        Benchmark.__init__(self, name=file)

        # Read queries as strings from file
        with open(file, 'r') as f:
            lines = list(f)

        # Load jsons from strings
        self._queries = [
            json.loads(x[:-1])
            for x in lines
        ]

    def _run(self, instance):
        """ Execute all queries in self._queries """
        [instance.execute(q) for q in self._queries]
