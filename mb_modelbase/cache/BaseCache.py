import abc
import dill


class BaseCache(abc.ABC):
    """An abstract baseclass for modelbase caches.

    Override the functions _get and _set to use various storage backends.

    Args:
        serialize (bool): sets if payload gets serialized before storage.
    Attributes:
        _serialize (bool): sets if payload gets serialized before storage.
    """

    def __init__(self, serialize=False):
        self._serialize = serialize

    def get(self, key, default=None):
        """Query the cache for the given key.

        This method looks up the key and returns the associated object if the key is known.
        If the key is not known the default is returned.

        Args:
            key: A representation of the object for unique identification.
            default: A default return value if key is not found.
            default if key is not found, else the payload stored for key.
        Returns:
            Data if key is known, default otherwise.
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
        """An abstract method to query cache for data.

        Args:
            key: A representation of the object for unique identification.

        Returns:
            data loaded from the storage backend.
        """

    def set(self, key, data):
        """A method to store data in the storage backend referenced by key

        Args:
            key: A representation of the object for unique identification
            data: Data to be associated with key and stored in the storage Backend
        """
        # If the cache is set to serialize, process data with dill
        if self._serialize:
            self._set(key, dill.dumps(data))
        else:
            self._set(key, data)

    @abc.abstractmethod
    def _set(self, key, data):
        """An abstract method to store data in the storage backend associated with the key

        Args:
            key: A representation of the object for unique identification
            data: Data to be associated with key and stored in the storage backend
        """
