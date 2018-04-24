# Copyright (c) 2017 Philipp Lucas (philipp.lucas@uni-jena.de)
import math

# TODO: is it better to use immutable tuples instead of mutable lists for the internal representation of domains?

# TODO: performance improvement
#  when modfiying a domain compute its state (i.e. bounded, singular, unbounded) and cache it
#  or: flexible caching on demand, i.e. cache on first request, reset cache on domain modification

# TODO: performance improvement
#  like above also cache a domains value() , values() and bounded()


class Domain:

    def apply(self, op, values):
        if op == 'in':
            self.intersect(values)
        else:
            # values is necessarily a single scalar value, not a list
            if op == 'equals' or op == '==':
                self.intersect(values)
            elif op == 'greater' or op == '>':
                self.setlowerbound(values)
            elif op == 'less' or op == '<':
                self.setupperbound(values)
            else:
                raise ValueError('invalid operator for condition: ' + str(op))


class NumericDomain(Domain):
    """A continuous domain that can be represented by an interval [min, max]."""

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

    def __str__(self):
        return str(self._value)

    def _validate(self):
        if self._value[0] > self._value[1]:
            raise ValueError("resulting domain is empty: " + str(self._value))

    def issingular(self):
        return self._value[0] == self._value[1]

    def isbounded(self):
        return self._value[0] != -math.inf and self._value[1] != math.inf

#    this actually is: is neither unbounded nor singular
#    def isbounded(self):
#        return not self.issingular() and not self.isunbounded()

    def bounded(self, extent, value_flag=False):
        if not self.isbounded():
            [l, h] = self._value
            try:
                [el, eh] = extent._value
            except AttributeError:
                [el, eh] = extent
            l = el if l == -math.inf else l
            h = eh if h == math.inf else h
            return [l, h] if value_flag else NumericDomain(l, h)
        else:
            return self.values() if value_flag else self

    def value(self):
        return self._value[0] if self.issingular() else self._value

    def values(self):
        return self._value

    def tojson(self):
        if self.isbounded():
            return self.value()
        else:
            [l, h] = self._value
            return [None if l == -math.inf else l, None if h == math.inf else h]

    def clamp(self, val):
        """Returns val clamped to the range of the domain. However, if val is None then None is returned."""
        if val is None:
            return None
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

    def mid(self):
        """Returns the mid element of the domain."""
        if not self.isbounded():
            raise ValueError("cannot compute mid of not bounded domain!")
        issingular = self.issingular()
        values = self.values()
        return values[0] if issingular else ((values[1] + values[0]) / 2)

    def contains(self, value):
        """Returns true iff value is an element of this domain."""
        return self._value[0] <= value <= self._value[1]


class DiscreteDomain(Domain):
    """An (ordered) discrete domain that can be represented by a list of values [val1, val2, ... ]."""

    def __init__(self, *args):
        """Constructs a discrete domain.
             * pass no arguments for an unbounded domain
             * not anymore: pass one scalar argument for a singular domain
             * pass a list of values for a bounded domain. its order is preserved.

           ONLY strings as categorical values are allowed!
        """

        """Internal representation is as follows:
            * self._value == [math.inf] for an unbound domain
            * self._value == [single_value] for a singular domain that only contains single_value
            * self._value == [val1, ... , valn] for a bounded domain of val1, ..., valn
        """
        l = len(args)
        if l == 0:
            self._value = [math.inf]
        elif l == 1:
            # convert to array if its a single value
            val = args[0]
            self._value = [val] if isinstance(val, str) else val  # implicitely assumes that values can only be strings
        else:
            raise ValueError("Too many arguments given: " + str(args))
        self._validate()

    def __len__(self):
        return math.inf if not self.isbounded() else len(self._value)

    def __str__(self):
        return str(self._value)

    def _validate(self):
        if len(self._value) == 0:
            raise ValueError("domain must not be empty")

    def issingular(self):
        return len(self._value) == 1 and self.isbounded()

    def isbounded(self):
        return self._value[0] != math.inf

    def bounded(self, extent, value_flag=False):
        if not self.isbounded():
            if isinstance(extent, DiscreteDomain):
                return extent.values() if value_flag else extent
            else:
                return extent if value_flag else DiscreteDomain(extent)
            # return extent if isinstance(extent, DiscreteDomain) else DiscreteDomain(extent)
        else:
            return self.values() if value_flag else self

    def value(self):
        return self._value[0] if self.issingular() else self._value

    def values(self):
        return self._value

    def tojson(self):
        # requires special treatment, because math.inf would often not be handled correctly in JSON
        if self.isbounded():
            return self.value()
        else:
            return None

    def clamp(self, val):
        """Returns val clamped to the range of the domain. However, if val is None then None is returned."""
        if val is None:
            return None
        if not self.isbounded() or val in self._value:
            return val
        else:
            raise NotImplementedError("Don't know what to do.")

    def intersect(self, domain):
        try:
            dvalue = domain._value
        except AttributeError:
            dvalue = DiscreteDomain(domain)._value

        if not self.isbounded():
            self._value = dvalue
        else:
            self._value = [e for e in self._value if e in dvalue]
        self._validate()
        return self

    def setlowerbound(self, value):
        raise NotImplementedError
        # use slice with find to find index to slice from

    def setupperbound(self, value):
        raise NotImplementedError

    def mid(self):
        """Returns the mid element of the domain. Since there is no ordering on the elements, there is no mid and
        a fixed but random element is returned."""
        if not self.isbounded():
            raise ValueError("cannot compute mid of not bounded domain!")
        values = self.values()
        return values[0]

    def contains(self, value):
        """Returns true iff value is an element of this domain."""
        return self._value[0] <= value <= self._value[1]