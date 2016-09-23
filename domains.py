import math

#class Domain:
#    pass


class NumericDomain:
    def __init__(self, *args):
        """Constructs a numercial domain.
             * pass no arguments for an unbounded domain
             * pass one scalar argument for a singular domain
             * pass a list/tuple of two scalars for a bounded or partially unbounded domain:
                [lower, upper], [val, +math.inf] or [-math.inf, val]
             * or pass two scalars
        """
        l = len(args)
        if l == 0:
            self._value = [-math.inf, +math.inf]
        elif l == 1:
            arg = args[0]
            try:
                self._value = [arg[0], arg[1]]
            except (TypeError, KeyError, IndexError):  # 'number' is not subscriptable
                self._value = [arg, arg]
        elif l == 2:
            self._value = [args[0], args[1]]
        else:
            raise ValueError("Too many arguments given: " + str(args))
        self._validate()

    def _validate(self):
        if self._value[0] > self._value[1]:
            raise ValueError("resulting domain is empty: " + str(self._value))

    def issingular(self):
        return self._value[0] == self._value[1]

    def isunbounded(self):
        return self._value[0] == -math.inf or self._value[1] == math.inf

    def isbounded(self):
        return not self.issingular() and not self.isunbounded()

    def bounded(self, extent):
        if self.isunbounded():
            [l, h] = self._value
            try:
                [el, eh] = extent._value
            except AttributeError:
                [el, eh] = extent
            return NumericDomain(el if l == -math.inf else l, eh if h == math.inf else h)
        else:
            return self

    def value(self):
        return self._value[0] if self.issingular() else self._value

    def tojson(self):
        if not self.isunbounded():
            return self.value()
        else:
            [l, h] = self._value
            return [None if l == -math.inf else l, None if h == math.inf else h]

    def clamp(self, val):
        [l, h] = self._value
        if val < l:
            return l
        elif val > h:
            return h
        return val

    def intersect(self, domain):
        try:
            [l2, h2] = domain._value
        except AttributeError:
            [l2, h2] = NumericDomain(domain)._value
        [l1, h1] = self._value

        self._value = [
            l1 if l1 > l2 else l2,
            h1 if h1 < h2 else h2
        ]
        self._validate()
        return self

    def setlowerbound(self, value):
        if value > self._value[0]:
            self._value[0] = value
        self._validate()
        return self

    def setupperbound(self, value):
        if value < self._value[1]:
            self._value[1] = value
        self._validate()
        return self

#class DiscreteDomain (Domain):




# def _issingulardomain(domain):
#     """Returns True iff the given domain is singular, i.e. if it _is_ a single value."""
#     return domain != math.inf and (isinstance(domain, str) or not isinstance(domain, (list, tuple)))  # \
#     # or len(domain) == 1\
#     # or domain[0] == domain[1]
#
#
# def _isboundeddomain(domain):
#     # its unbound if domain == math.inf or if domain[0] or domain[1] == math.inf
#     if domain == math.inf:
#         return False
#     if _issingulardomain(domain):
#         return True
#     l = len(domain)
#     if l > 1 and (domain[0] == math.inf or domain[1] == math.inf):
#         return False
#     return True
#
# def _isunbounddomain(domain):
#     return not _issingulardomain(domain) and not _isboundeddomain(domain)
#
# def _boundeddomain(domain, extent):
#     if domain == math.inf:
#         return extent  # this is the only case a ordinal domain is unbound
#     if _issingulardomain(domain):
#         return domain
#     if len(domain) == 2 and (
#             domain[0] == -math.inf or domain[1] == math.inf):  # hence this fulfills only for cont domains
#         low = extent[0] if domain[0] == -math.inf else domain[0]
#         high = extent[1] if domain[1] == math.inf else domain[1]
#         return [low, high]
#     return domain
#
#
# def _clamp(domain, val, dtype):
#     if dtype == 'string' and val not in domain:
#         return domain[0]
#     elif dtype == 'numerical':
#         if val < domain[0]:
#             return domain[0]
#         elif val > domain[1]:
#             return domain[1]
#     return val
#
#
# def _jsondomain(domain, dtype):
#     """"Returns a domain that can safely be serialized to json, i.e. any infinity value is replaces by null."""
#     if not _isunbounddomain(domain):
#         return domain
#     if dtype == 'numerical':
#         l = domain[0]
#         h = domain[1]
#         return [None if l == -math.inf else l, None if h == math.inf else h]
#     elif dtype == 'string':
#         return None
#     else:
#         raise ValueError('invalid dtype of domain: ' + str(dtype))
#
