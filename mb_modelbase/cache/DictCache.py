from mb_modelbase.cache import BaseCache


class DictCache(BaseCache):
    def __init__(self):
        BaseCache.__init__(self, serialize=False)
        self.cache = {}

    def _get(self, key):
        return self.cache.get(key, None)

    def _set(self, key, model):
        self.cache[key] = model

    def keys(self):
        return self.cache.keys()
