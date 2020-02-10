from mb_modelbase.utils.benchmark.time_ModelbaseCache import Benchmark
import mb_modelbase.cache.cache as mc
from mb_modelbase.server import modelbase as mbase
import time
import pandas as pd
import sys
import plotly.express as px

class ModelSerializationBechmark(Benchmark):
    def __init__(self):
        Benchmark.__init__('ModelSerializationBechmark')



if __name__ == "__main__":
    cache = mc.DictCache()

    mbase = mbase.ModelBase(
        name='dict_cache',
        model_dir='/home/leng_ch/git/lumen/fitted_models',
        cache=mc.DictCache())

    def models():
        return [m.copy() for m in mbase.models.values()]

    def timeSer(model,n = 100):
        start = time.time()
        for i in range(n):
            cache.set('test', model)
            res = cache.get('test')
        end = time.time()
        return (end-start) / n

    names = [m.name for m in models()]
    withData = models()

    withoutData = models()
    for m in withoutData:
        m.data = None

    df = pd.DataFrame(
        {
            'model': names,
            'withData': [timeSer(m) for m in withData],
            'withoutData': [timeSer(m) for m in withoutData],
            'dataSize': [sys.getsizeof(m.data) for m in withData]
            #'withoutDataSize': [sys.getsizeof(m.data) for m in withoutData]
        }
    )


    # Plot
    fig = px.bar(df.drop('dataSize', 1).melt(id_vars='model'), x='model', y='value',barmode='group',
                  color='value',
                 labels={'pop': 'population of Canada'}, height=400)

    fig.write_image('/home/leng_ch/git/lumen_caching/presentation/fig/serial.png')

    #fig.show()
