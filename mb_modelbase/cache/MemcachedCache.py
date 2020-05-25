from mb_modelbase.cache import BaseCache
from pymemcache.client import base


class MemcachedCache(BaseCache):
    """A cache with memcached as storage backend.

    Attributes:
        _hostname (str): hostname of memcached server.
        _port (int): port of memcached server.
        _client (base.Client): The memcached client.
    Args:
        hostname (str): hostname of memcached server.
        port (int): port of memcached server.
    """

    def __init__(self,
                 hostname='0.0.0.0',
                 port=11211):
        BaseCache.__init__(self)
        self._hostname = hostname
        self._port = port
        self._client = base.Client((self._hostname, self._port))

    def _get(self, key, default=None):
        return self._client.get(key)

    def _set(self, key, data):
        self._client.set(key, data)

