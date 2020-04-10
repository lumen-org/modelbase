from mb_modelbase.server import ModelBase
from mb_modelbase.utils.benchmark import ModelbaseActivityfileBenchmark
import pandas as pd
from mb_modelbase import DictCache
import os

"""This script performs the ModelactivityfileBenchmark on two modelbases
One the modelbases with no cache, one with a DictCache
"""

if __name__ == '__main__':
    model_dir = os.path.expanduser("~/git/lumen/fitted_models")
    no_cache = ModelBase(
        name='no_cache',
        model_dir=model_dir,
        cache=None)

    dict_cache = ModelBase(
        name='dict_cache',
        model_dir=model_dir,
        cache=DictCache())

    # mem_cache = mbase.ModelBase(
    #     name='no_cache',
    #     model_dir='../../../fitted_models',
    #     cache=mc.MemcachedCache())

    bases = [
        no_cache,
        dict_cache  # ,
        # mem_cache
    ]

    # Disable parallel processing to enable profiling with cProfile/snakeviz
    for m in dict_cache.models.values():
        m.parallel_processing = False

    benchmarks = [
        ModelbaseActivityfileBenchmark
    ]

    df = pd.DataFrame({base.name: [benchmark('interactionPhilippMod.log').run(
        base) for benchmark in benchmarks] for base in bases})

    # df.to_csv(os.path.expanduser("~/git/lumen_caching/data/modelbaseCache_" + time.strftime("%b:%d:%Y_%H:%M:%S", time.gmtime(time.time())) + ".csv"))
