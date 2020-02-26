import abc
import dill

def modelKey(name, model, where) -> str:
    # Use naive key for testing
    return (str(name) + ':' + str(model).strip('[]') + ':' + str(where).strip('[]')).replace(' ', '')

def predictKey(base, predict_stmnt, where_stmnt, splitby_stmnt) -> str:
    return str(base) + str(predict_stmnt) + str(where_stmnt) + str(splitby_stmnt)

class BaseCache(abc.ABC):
    def __init__(self,serialize=False):
        self._serialize = serialize

    def get(self, key, default=None):
        data = self._get(key)
        if data is None:
            # Make sure the default is return if key is not in Cache
            return default
        else:
            if self._serialize:
                return dill.loads(data)
            else:
                return data

    @abc.abstractmethod
    def _get(self, key):
        """ Check if cache has key, return item for key or None """

    def set(self, key, model):
        if self._serialize:
            self._set(key, dill.dumps(model))
        else:
            self._set(key, model)

    @abc.abstractmethod
    def _set(self, key, value):
        """ store value with key """

    @abc.abstractmethod
    def keys(self):
        """ Get all known keys """
