from abc import ABCMeta


class BaseObjective(metaclass=ABCMeta):
    """ Base class to implement optimization objectives used in Streaming Gradient Trees. """

    def compute_derivatives(self, y: dict, y_pred: dict):
        """ Return the gradient and hessian data concerning one instance and its prediction.

        Parameters
        ----------
        y
            Dictionary in the form (target_idx, target_value) containing the target(s)' ground
            truth.
        y_pred
            Dictionary in the form (target_idx, target_value) containing the target(s)'
            predicted value(s).
        """
        pass

    def transfer(self, y: dict):
        """ Optionally apply some transformation to the values predicted by the trees before
        returning them.

        For instance, in classification, the softmax operation is applied.

        Parameters
        ----------
        y
            Values to be transformed
        """
        return y
