from abc import ABCMeta
import math
from bisect import bisect_left


class BaseEBFQStat(metaclass=ABCMeta):
    """ Base class concerning the data monitored by the EqualBinFeatureQuantizer (EBFQ).

    Defines basic common operations shared accross different kinds of monitored data in the
    Locality Sensitive Hashing strategy used in EBFQ.
    """
    def get_x(self):
        pass

    def __add__(self, stat):
        pass

    def __getitem__(self, key):
        pass

    def clone(self):
        pass


class EqualBinFeatureQuantizer:
    """ Unidimensional Locality Sensitive Hashing implementation used to stored statistics. """
    def __init__(self, radius):
        self.radius = radius
        # Define the random projection

        self.hash = {}
        self.indexes = []
        self._min_val = float('Inf')
        self._min_idx = None

    def __getitem__(self, k):
        return self.hash[k]

    def __len__(self):
        return len(self.hash)

    def add(self, stat):
        index = math.floor(stat.get_x() / self.radius)
        if index in self.hash:
            self.hash[index] += stat
        else:
            x = stat.get_x()
            self.hash[index] = stat
            pos = bisect_left(self.indexes, index)
            self.indexes.insert(pos, index)

            if x < self._min_val:
                self._min_val = x
                self._min_idx = pos
            elif pos <= self._min_idx:  # Shift index of the smallest element's bin
                self._min_idx += 1

    def ordered(self):
        """ Generator that enables an ordered pass over the LSH structure. """
        start = self._min_idx

        for i in range(start, len(self.indexes)):
            yield self.indexes[i]

        for i in range(start):
            yield self.indexes[i]

    def ordered_rev(self):
        """ Generator that enables a reverse ordered pass over the LSH structure. """
        start = self._min_idx - 1
        if start == -1:
            start = len(self.hash) - 1

        for i in range(start, -1, -1):
            yield self.indexes[i]

        for i in range(len(self.hash) - 1, start, -1):
            yield self.indexes[i]
