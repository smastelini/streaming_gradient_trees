import math

from .base_objective import BaseObjective
from ..utils import GradHess


class BinaryCrossEntropyObjective(BaseObjective):
    """ Loss function used in multi-label classification tasks. """
    def compute_derivatives(self, y, y_pred):
        pred = self.transfer(y_pred)

        return GradHess(pred - y, pred * (1.0 - pred))

    def transfer(self, y):
        return 1.0 / (1.0 + math.exp(-y))
