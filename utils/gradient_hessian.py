from .equal_bin_feature_quantizer import BaseEBFQStat


class GradHess:
    """ The most basic inner structure of the Streaming Gradient Trees. """
    def __init__(self, gradient=0.0, hessian=0.0, *, grad_hess=None):
        if grad_hess is not None:
            self.gradient = grad_hess.gradient
            self.hessian = grad_hess.hessian
        else:
            self.gradient = gradient
            self.hessian = hessian

    def __add__(self, gh):
        self.gradient += gh.gradient
        self.hessian += gh.hessian

        return self

    def __sub__(self, gh):
        self.gradient -= gh.gradient
        self.hessian -= gh.hessian

        return self

    def clone(self):
        return GradHess(grad_hess=self)


class GradHessStats(BaseEBFQStat):
    """ Class used to monitor and update the gradient/hessian information in Streaming Gradient
    Trees.

    Represents the aggregated gradient/hessian data in a node (global node statistics), category,
    or numerical feature discretized bin.
    """
    def __init__(self):
        self._sum_x = 0.0

        self._sum = GradHess()
        self._scaled_var = GradHess()
        self._scaled_cov = 0.0
        self._n_observations = 0

    def get_x(self):
        """ Get the centroid x data that represents all the observations inside a bin. """
        if self._n_observations > 0.0:
            return self._sum_x / self._n_observations

        return None

    def __add__(self, stats):
        if stats._n_observations == 0:
            return self

        if self._n_observations == 0:
            self._sum = GradHess(grad_hess=stats._sum)
            self._scaled_var = GradHess(grad_hess=stats._scaled_var)
            self._scaled_cov = stats._scaled_cov
            self._n_observations = stats._n_observations

            return self

        mean_diff = stats.mean()
        mean_diff -= self.mean()
        n1 = self._n_observations
        n2 = stats._n_observations

        # Scaled variance (see Wikipedia page on "Algorithms for calculating variance", section
        # about parallel calculation)
        self._scaled_var.gradient += (stats._scaled_var.gradient + mean_diff.gradient *
                                      mean_diff.gradient * (n1 * n2) / (n1 + n2))
        self._scaled_var.hessian += (stats._scaled_var.hessian + mean_diff.hessian *
                                     mean_diff.hessian * (n1 * n2) / (n1 + n2))

        # Scaled covariance (see "Numerically Stable, Single-Pass, Parallel Statistics Algorithms"
        # (Bennett et al, 2009))
        self._scaled_cov += (stats._scaled_cov + mean_diff.gradient * mean_diff.hessian *
                             (n1 * n2) / (n1 + n2))

        self._sum += stats._sum
        self._n_observations += stats._n_observations

        return self

    def add_instance(self, grad_hess, x=None):
        # Update x values in the case of numerical features (binning strategy)
        if x is not None:
            self._sum_x += x

        old_mean = self.mean()
        self._sum += grad_hess
        self._n_observations += 1.0
        new_mean = self.mean()

        self._scaled_var.gradient += (grad_hess.gradient - old_mean.gradient) * \
            (grad_hess.gradient - new_mean.gradient)
        self._scaled_var.hessian += (grad_hess.hessian - old_mean.hessian) * \
            (grad_hess.hessian - new_mean.hessian)
        self._scaled_cov += (grad_hess.gradient - old_mean.gradient) * \
            (grad_hess.hessian - new_mean.hessian)

    def mean(self):
        if self._n_observations == 0:
            return GradHess()
        else:
            return GradHess(self._sum.gradient / self._n_observations,
                            self._sum.hessian / self._n_observations)

    def variance(self):
        if self._n_observations < 2:
            return GradHess()
        else:
            return GradHess(self._scaled_var.gradient / (self._n_observations - 1),
                            self._scaled_var.hessian / (self._n_observations - 1))

    def covariance(self):
        if self._n_observations < 2:
            return 0.0
        else:
            return self._scaled_cov / (self._n_observations - 1)

    @property
    def n_observations(self):
        return self._n_observations

    def delta_loss_mean(self, delta_pred):
        m = self.mean()

        return delta_pred * m.gradient + 0.5 * m.hessian * delta_pred * delta_pred

    # This method ignores correlations between delta_pred and the gradients/hessians! Considering
    # delta_pred is derived from the gradient and hessian sample, this assumption is definitely
    # violated. However, as empirically demonstrated on the original SGT, this fact does not seem
    # to significantly impact on the obtained results.
    def delta_loss_variance(self, delta_pred):
        variance = self.variance()
        covariance = self.covariance()

        grad_term_var = delta_pred * delta_pred * variance.gradient
        hess_term_var = 0.25 * variance.hessian * (delta_pred ** 4.0)

        return max(0.0, grad_term_var + hess_term_var + (delta_pred ** 3) * covariance)

    def clone(self):
        new = GradHessStats()
        new._sum_x = self._sum_x

        new._sum = self._sum.clone()
        new._scaled_var = self._scaled_var.clone()
        new._scaled_cov = self._scaled_cov
        new._n_observations = self._n_observations

        return new

    @staticmethod
    def combine_mean(m1, n1, m2, n2):
        if n1 == 0:
            return m2
        if n2 == 0:
            return m1
        return (m1 * n1 + m2 * n2) / (n1 + n2)

    @staticmethod
    def combine_variance(m1, s1, n1, m2, s2, n2):
        if n1 == 0:
            return s2
        if n2 == 0:
            return s1

        n = n1 + n2
        m = GradHessStats.combine_mean(m1, n1, m2, n2)

        # First we have to bias the sample variances (we'll unbias this later)
        s1 = ((n1 - 1) / n1) * s1
        s2 = ((n2 - 1) / n2) * s2

        # Compute the sum of squares of all the datapoints
        t1 = n1 * (s1 + m1 * m1)
        t2 = n2 * (s2 + m2 * m2)
        t = t1 + t2

        # Now get the full (biased) sample variance
        s = t / n - m

        # Apply Bessel's correction
        s = (n / (n - 1)) * s

        return s
