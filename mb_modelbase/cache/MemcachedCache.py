from mb_modelbase.cache import BaseCache
from pymemcache.client import base


class MemcachedCache(BaseCache):
    """
    A cache with memcached as storage backend

    Attributes:
        hostname (str): hostname of memcached server
        port (int): port of memcached server
    """

    def __init__(self,
                 hostname='0.0.0.0',
                 port=11211):
        BaseCache.__init__(self)
        self._hostname = hostname
        self._port = port
        self._keys = {}
        self._client = base.Client((self._hostname, self._port))

    def _get(self, key, default=None):
        return self._client.get(key)

    def _set(self, key, data):
        self._keys[key] = True
        self._client.set(key, data)

    def keys(self):
        return self._keys.keys()
