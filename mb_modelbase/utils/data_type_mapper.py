# Copyright (c) 2019 Philipp Lucas (philipp.lucas@dlr.de)
"""
@author: Philipp Lucas
"""
import pandas as pd
import copy

class DataTypeMapper:
    """ Maps data between two representations by user defined per dimension mappings.

    Any missing map will act as the identity mapping.
    """

    def __init__(self):
        self._forward_map = {}
        self._backward_map = {}

    def forward(self, data, inplace=True):
        """Map data from original into alternative representation.
        TODO: allow to receive only a list of values, where values are in correct order
        Args:
            data: pd.DataFrame, pd.Series, dict.
                Data to map.
            inplace: bool, defaults to True.
                If True the mapping is done inplace, otherwise a copy is returned.
        """
        return self._map(data, self._forward_map, inplace)

    def backward(self, data, inplace=True):
        """Map data from alternative into original representation.
        Args:
            data: pd.DataFrame, pd.Series, dict.
                Data to map.
            inplace: bool, defaults to True.
                If True the mapping is done inplace, otherwise a copy is returned.
        """
        return self._map(data, self._backward_map, inplace)

    def _map(self, data, mapping, inplace):
        obj_type = type(data)
        if obj_type is pd.DataFrame:
            if inplace is False:
                data = data.copy()
            columns = data.columns.intersection(mapping.keys())
            sub_mapping = {c: mapping[c] for c in columns}
            data.replace(to_replace=sub_mapping, inplace=True)
            return data

        elif obj_type is pd.Series:
            if inplace is False:
                data = data.copy()
            data.replace(to_replace=mapping[data.name], inplace=True)
            return data

        elif obj_type is dict:
            # replace each dict value individually
            mapping_keys = set(mapping.keys())
            data_mapped = {name: mapping[name][value] if name in mapping_keys else value
                           for name, value in data.items()}
            if inplace is True:
                data.update(data_mapped)
            else:
                data = data_mapped
            return data

    def set_map(self, name, forward=None, backward=None, mode='single'):
        """Set mappings for a variable.
        TODO: implement vector mode?
        Args:
            name: str
                Name of variable
            forward: dict or callable, optional. Defaults to None.
                Callable that convert a single data item (if mode is 'single') or a iterable of data (if mode is
                'vector') items into the alternative representation.
                If None the forward mapping is left unchanged.
            backward: dict or callable, optional. Defaults to None.
                Callable that convert a single data item (if mode is 'single') or a iterable of data items from the
                 alternative representation into the original representation.
                 If None the backward mapping is left unchanged.
            mode: str, one of ['single', 'vector']
                Select the mode to set the mapping for.
        """
        if mode is not 'single':
            raise NotImplementedError("mode different from 'single' is not yet implemented.")
        if forward is 'auto':
            if type(backward) is dict:
                forward = DataTypeMapper.invert_dict_mapping(backward)
            else:
                raise TypeError('backward has to be dict if forward is set to auto.')
        if backward is 'auto':
            if type(forward) is dict:
                backward = DataTypeMapper.invert_dict_mapping(forward)
            else:
                raise TypeError('forward has to be dict if backward is set to auto.')
        self._forward_map[name] = forward
        self._backward_map[name] = backward

    def copy(self):
        copy_ = DataTypeMapper()
        copy_._forward_map = copy.deepcopy(self._forward_map)
        copy_._backward_map = copy.deepcopy(self._backward_map)
        return copy_

    @staticmethod
    def invert_dict_mapping(dict_mapping):
        """Invert the dictionary mapping and return it.

        Args:
            dict_mapping: dict
                A mapping of key to values.

        Raises:
            ValueError if values in dict are not unique
        """
        keys = dict_mapping.keys()
        values = dict_mapping.values()
        if len(values) != len(set(values)):
            raise ValueError("inversion impossible as values in dict are not unique")
        return dict(zip(values, keys))


if __name__ == '__main__':
    pass
