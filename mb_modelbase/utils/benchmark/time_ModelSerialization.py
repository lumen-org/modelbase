from mb_modelbase.cache import DictCache
from mb_modelbase import ModelBase
from mb_modelbase.utils.benchmark import ModelSerializationBenchmark
import time
import pandas as pd
import sys
import os
from pympler import asizeof

"""
This script performs the ModelSerialization Benchmark on a modelbase with a DictCache
"""

if __name__ == "__main__":
    cache = DictCache()
    b = ModelSerializationBenchmark(cache=cache)

    mbase = ModelBase(
        name='dict_cache',
        model_dir=os.path.expanduser('~/git/lumen/fitted_models'),
        cache=cache)


    def models():
        return [m.copy() for m in mbase.models.values()]


    withData = models()

    withoutData = models()
    for m in withoutData:
        m.data = None

    df = pd.DataFrame(
        {
            'model': [m.name for m in models()],
            'withData': [b.run(m) for m in withData],
            'withoutData': [b.run(m) for m in withoutData],
            'dataSize': [asizeof.asizeof(m) for m in withData]
            # 'withoutDataSize': [sys.asizeof(m.data) for m in withoutData]
        }
    )
    # df.to_csv(os.path.expanduser("~/git/lumen_caching/data/modelSerialization") + time.strftime("%b:%d:%Y_%H:%M:%S", time.gmtime(time.time())) + ".csv")
