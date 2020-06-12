from abc import ABCMeta


class BaseObjective(metaclass=ABCMeta):
    """ Base class to implement optimization objectives used in Streaming Gradient Trees. """

    def compute_derivatives(self, y, y_pred):
        """ Return the gradient and hessian data concerning one instance and its prediction.

        Parameters
        ----------
            y: dict
                Dictionary in the form (target_idx, target_value) containing the target(s)' ground
                truth.
            y_pred: dict
                Dictionary in the form (target_idx, target_value) containing the target(s)'
                predicted value(s).
        """
        pass

    def transfer(self, y):
        """ Optionally apply some transformation to the values predicted by the trees before
        returning them.

        For instance, in classification, the softmax operation is applied.
        """
        return y
