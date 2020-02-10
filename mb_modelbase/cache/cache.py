import abc
from pymemcache.client import base
import dill

def key(name, model,where):
    # Use naive key for testing
    return (str(name) + ':' + str(model).strip('[]') + ':' + str(where).strip('[]')).replace(' ', '')

class CacheBase(abc.ABC):
    def get(self, key, default=None):
        model = self._get(key, default)
        if model == default:
            # print("CACHE MISS!")
            return model
        else:
            #print("CACHE HIT!")
            return dill.loads(model)

    @abc.abstractmethod
    def _get(self, key, default=None):
        """ Check if cache has key """

    def set(self, key, model):
        self._set(key, dill.dumps(model))

    @abc.abstractmethod
    def _set(self, key, value):
        """ store value with key """

    @abc.abstractmethod
    def keys(self):
        """ Get all known keys """

class DictCache(CacheBase):
    def __init__(self):
        CacheBase.__init__(self)
        self.cache = {}

    def _get(self, key, default=None):
        return self.cache.get(key,default)

    def _set(self, key, model):
        self.cache[key] = model

    def keys(self):
        return self.cache.keys()

class MemcachedCache(CacheBase):
    def __init__(self,
                 hostname='0.0.0.0',
                 port=11211):
        CacheBase.__init__(self)
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

if __name__ == '__main__':
    c = DictCache()
    c.set('bla',5)
    print(c.keys())
    print(c.get('bla'))

    c = base.Client(('localhost', 11212))

    mc = MemcachedCache()
    mc.set('bla', 'blub')
