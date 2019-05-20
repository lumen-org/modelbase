import abc
from pymemcache.client import base as memcBase

class Cache():
    def __init__(self, *args, **kwargs):
        """Initialize Connections"""

    @abc.abstractmethod
    def get(self, key, default=None):
        """ returns object for key or None"""

    @abc.abstractmethod
    def set(self, key, value):
        """ Stores object with given key"""


class DictCache(Cache):
    def __init__(self, *args, **kwargs):
        Cache.__init__(*args, **kwargs)
        self._cache = {}

    def set(self, key, value):
        self._cache.set(key, value)

    def get(self, key, default):
        return self._cache.get(key,default)

class MemcachedCache(Cache):
    def __init__(self, *args, **kwargs):
        Cache.__init__(*args, **kwargs)

        self._expire_time = kwargs.get('memcached_expire_time', 0)

        self._cache = memcBase.Client((
            kwargs.get('memcached_hostname', 'localhost'),
            kwargs.get('memcached_port', 11211),
        ))

    def set(self, key, value):
        self._cache.set(
            key,
            value,
            self._expire_time
        )

    def get(self, key, default = None):
        return self._cache.get(key)

if __name__ == '__main__':
    memCache = MemcachedCache({
        'memcached_hostname' : 'localhost',
        'memcached_port' : '11211'
    })

    # Try large cache
    [memCache.set( str(x), True) for x in range(1000)]