from mb_modelbase.cache import BaseCache

class MemcachedCache(BaseCache):
    def __init__(self,
                 hostname='0.0.0.0',
                 port=11211):
        BaseCache.__init__(self)
        self.hostname = hostname
        self.port = port
        self.keys = {}
        self.client = base.Client((self.hostname, self.port))

    def _get(self, key, default=None):
        return self.client.get(key)

    def _set(self, key, model):
        self.keys[key] = True
        self.client.set(key, model)

    def keys(self):
        return self.keys.keys()
