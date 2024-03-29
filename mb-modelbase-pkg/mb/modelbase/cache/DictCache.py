import logging
import time
import pathlib
import pickle
import threading
import os

from mb.modelbase.cache import BaseCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DictCache(BaseCache):
    """Cache with a regular python dictionary as a backend.
    The cache stores the content after every save_interval to a file on disk.
    Mutual exclusion is ensured with a lock.

    Attributes:
        cache_dir (str): The path to where the cache should save the dictionary.
        save_interval (float): The interval in seconds on which to save the content of the dictionary.
        lock (threading.Lock): A lock for mutual exclusion.
    Args:
        cache_dir (str): The path to where the cache should save the dictionary.
        save_interval (float): The interval in seconds on which to save the content of the dictionary.

    Todo:
        * Implement cache as a monitor like in the readers writers problem
        * Maybe lift mutual exclusion to superclass?
        * Make name of cache file unique
        * Make writing out of cache more performant and asynchronous
    """

    def __init__(self, cache_dir='/tmp', save_interval=30):
        BaseCache.__init__(self, serialize=False)

        if not os.path.exists(cache_dir):
            try:
                pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
            except IOError:
                raise IOError('The base directory for the cache does not exists and can not be'
                              ' created automatically.: {}'.format(str(cache_dir)))
            else:
                logger.info('created new base directory for cache at {}'.format(str(cache_dir)))

        self.lock = threading.Lock()
        self.save_path = os.path.abspath(cache_dir) + '/modelbaseCache'
        self.save_interval = save_interval

        # Initialize cache from file if it exists
        try:
            f = open(self.save_path, "rb")
        except IOError:
            self.cache = {}
        else:
            with f:
                self.cache = pickle.load(f)

        # Set time for saving to disk
        self.nextSave = time.time() + save_interval

    def _get(self, key: str):
        self.lock.acquire()
        res = self.cache.get(key, None)
        self.lock.release()
        return res

    def _set(self, key: str, data):
        self.lock.acquire()
        now = time.time()
        self.cache[key] = data

        # If save_interval has passed, save to disk
        if self.nextSave < now:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.cache, f)
            self.nextSave = now + self.save_interval
        self.lock.release()

    def keys(self):
        return self.cache.keys()
