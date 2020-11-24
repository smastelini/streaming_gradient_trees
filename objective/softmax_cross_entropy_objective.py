import math

from river.utils.math import softmax

from .base_objective import BaseObjective
from ..utils import GradHess


class SoftmaxCrossEntropyObjective(BaseObjective):
    """ Loss function used in binary and multiclass classification problems. """
    def compute_derivatives(self, y, y_pred):
        result = {}

        preds = self.transfer(y_pred)

        for class_idx in y:
            pred = preds.get(class_idx, 0)  # Default to 0 in case of missing class/tree
            result[class_idx] = GradHess(pred - y[class_idx], pred * (1.0 - pred))

        return result

    def transfer(self, y):
        return softmax(y)
