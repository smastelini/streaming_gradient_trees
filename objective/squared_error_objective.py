# TODO: fix that
import sys

sys.path.append('..')

from .base_objective import BaseObjective

from utils import GradHess


class SquaredErrorObjective(BaseObjective):
    """ Loss function used in regression tasks. """
    def compute_derivatives(self, y, y_pred):
        result = {}

        for k in y:
            # In case target is missing in predictions, default to zero
            result[k] = GradHess(y_pred.get(k, 0.0) - y[k], 1.0)

        return result
