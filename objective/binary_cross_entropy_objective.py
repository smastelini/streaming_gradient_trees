import math

from .base_objective import BaseObjective
from ..utils import GradHess


class BinaryCrossEntropyObjective(BaseObjective):
    """ Loss function used in multi-label classification tasks. """
    def compute_derivatives(self, y, y_pred):
        result = {}

        preds = self.transfer(y_pred)

        for label_idx in y:
            pred = preds.get(label_idx, 0)  # Defaults to 0 in case of missing class/tree
            result[label_idx] = GradHess(pred - y[label_idx], pred * (1.0 - pred))

        return result

    def transfer(self, y):
        result = {}
        for label_idx in y:
            result[label_idx] = 1.0 / (1.0 + math.exp(-y[label_idx]))

        return result
