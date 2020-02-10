import abc
import dill

class BaseCache(abc.ABC):
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
