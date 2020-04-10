import abc
import dill
from typing import List


def model_key(name: str, model: List[str], where: List[str]) -> str:
    """Function that computes a key for a given model.

    :param name:
    :param model:
    :param where:
    :return:
    """

    return (
        str(name) +
        ':' +
        str(model).strip('[]') +
        ':' +
        str(where).strip('[]')).replace(
        ' ',
        '')


def predict_key(
        base,
        predict_stmnt: List[str],
        where_stmnt,
        splitby_stmnt) -> str:
    """Function that computes a key for a given prediction query.

    :param base:
    :param predict_stmnt:
    :param where_stmnt:
    :param splitby_stmnt:
    :return:
    """

    return str(base) + str(predict_stmnt) + \
        str(where_stmnt) + str(splitby_stmnt)


class BaseCache(abc.ABC):
    """An abstract baseclass for modelbase caches.

    Override the functions _get and _set to use various storage backends.

    Attributes:
        _serialize (bool): sets if payload gets serialized before storage
    """

    def __init__(self, serialize=False):
        self._serialize = serialize

    def get(self, key, default=None):
        """Searches stored object for a given key, returns object or default


        :param key: A representation of the object for unique identification
        :param default: A default return value if key is not found.
        :return: default if key is not found, else the payload stored for key
        """
        # Check if storage has object for key
        data = self._get(key)
        # If nothing is found return the default value
        if data is None:
            return default
        else:
            if self._serialize:
                return dill.loads(data)
            else:
                return data

    @abc.abstractmethod
    def _get(self, key):
        """Abstract function to query cache for objects

        :param key: A representation of the object for unique identification
        :return: object loaded from the storage backend
        """

    def set(self, key, data):
        """Save data in the storage backend referenced by key

        :param key: A representation of the object for unique identification
        :param data: Data to be associated with key and stored in the storage Backend
        """
        # If cache is set to serialize, process data with dill
        if self._serialize:
            self._set(key, dill.dumps(data))
        else:
            self._set(key, data)

    @abc.abstractmethod
    def _set(self, key, data):
        """Abstract function to store values in the storage backend referenced by the key

        :param key: A representation of the object for unique identification
        :param data: Data to be associated with key and stored in the storage Backend
        :return:
        """
