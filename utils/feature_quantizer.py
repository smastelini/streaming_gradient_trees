import math


class FeatureQuantizer:
    """ Adapted version of the Quantizer Observer (QO) that is applied to SGTs. """
    def __init__(self, radius):
        self.radius = radius
        # Define the random projection
        self.hash = {}

    def __getitem__(self, k):
        return self.hash[k]

    def __len__(self):
        return len(self.hash)

    def update(self, ghs):
        x = ghs.get_x()
        index = math.floor(x / self.radius)
        if index in self.hash:
            self.hash[index] += ghs
        else:
            self.hash[index] = ghs

    def __iter__(self):
        for k in sorted(self.hash):
            yield self.hash[k]
