from mb_modelbase.cache import BaseCache

class DictCache(BaseCache):
    def __init__(self):
        BaseCache.__init__(self)
        self.cache = {}

    def _get(self, key, default=None):
        return self.cache.get(key,default)

    def _set(self, key, model):
        self.cache[key] = model

    def keys(self):
        return self.cache.keys()
