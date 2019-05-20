import abc
from pymemcache.client import base as memcBase
import redis

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
        Cache.__init__(self,*args, **kwargs)
        self._cache = {}

    def set(self, key, value):
        self._cache[key] = value

    def get(self, key, default=None):
        return self._cache.get(key, default)

class MemcachedCache(Cache):
    def __init__(self, *args, **kwargs):
        Cache.__init__(*args, **kwargs)

        self._expire_time = kwargs.get('memcached_expire_time', 0)


        self._cache = memcBase.Client((
            kwargs.get('hostname', 'localhost'),
            kwargs.get('port', 11211),
        ))

    def set(self, key, value):
        self._cache.set(
            key,
            value,
            self._expire_time
        )

    def get(self, key, default=None):
        return self._cache.get(key)

class RedisCache(Cache):
    def __init__(self, *args, **kwargs):
        Cache.__init__(*args, **kwargs)

        self._cache = redis.Redis(
            host=kwargs.get('hostname', 'localhost'),
            port=kwargs.get('port', 6379),
            db=kwargs.get('redis_db', 0)
        )

    def set(self, key, value):
        self._cache.set(
            key,
            value,
        )

    def get(self, key, default=None):
        return self._cache.get(key)

if __name__ == '__main__':

    dictCache = DictCache()
    [dictCache.set(str(x), True) for x in range(1000)]
    res = [dictCache.get(str(x)) for x in range(1000)]

    # docker run --name memcached -d memcached:alpine
    memCache = MemcachedCache({
        'hostname': 'localhost',
        'port': 11211
    })

    [memCache.set(str(x), str(x)) for x in range(1000)]
    res = [memCache.get(str(x)) for x in range(1000)]

    # docker run --name lumen_redis_cache -d redis
    redisCache = RedisCache({
        'hostname': 'localhost',
        'port': 6379
    })

    [redisCache.set( str(x), str(x)) for x in range(1000)]
    res = [redisCache.get(str(x)) for x in range(1000)]