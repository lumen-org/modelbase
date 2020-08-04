# Copyright (c) 2019-2020 Philipp Lucas (philipp.lucas@dlr.de)
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
            column_names = data.columns.intersection(mapping.keys())
            for c in column_names:
                self._map(data.loc[:, c], mapping, inplace=True)
            return data

        elif obj_type is pd.Series:
            if inplace is False:
                data = data.copy()
            series_mapping = mapping[data.name]
            if type(series_mapping) is dict:
                data.replace(to_replace=series_mapping, inplace=True)
            else:
                data.map(series_mapping)
            return data

        elif obj_type is dict:
            mapping_keys = set(mapping.keys())

            if inplace is True:
                # determine keys to update
                mapping_keys.intersection_update(data.keys())
                zipped = [(name, data[name], mapping[name]) for name in mapping_keys]

                update_dict = {name: (mapping(value) if callable(mapping) else mapping[value]) for name, value, mapping
                               in zipped}
                data.update(update_dict)
            else:
                data = {name: value if name not in mapping_keys else
                              (mapping[name](value) if callable(mapping[name]) else mapping[name][value])
                        for name, value in data.items()}
            return data

    def set_map(self, name, forward=None, backward=None):
        """Set mappings for a variable.
        Args:
            name: str
                Name of variable
            forward: dict or callable, optional. Defaults to None.
                Callable that convert a single data item into the alternative representation.
                If None the forward mapping is left unchanged.
            backward: dict or callable, optional. Defaults to None.
                Callable that convert a single data item into the original representation.
                 If None the backward mapping is left unchanged.
        """
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

        if forward is not None:
            if type(forward) is dict:
                self._forward_map[name] = forward
            else:
                raise NotImplementedError("callables as mappings not yet implemented")
        if backward is not None:
            if type(backward) is dict:
                self._backward_map[name] = backward
            else:
                raise NotImplementedError("callables as mappings not yet implemented")

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
