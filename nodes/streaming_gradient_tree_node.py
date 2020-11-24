import sys
import numbers

from ..utils import EqualBinFeatureQuantizer, GradHessStats


class SGTSplit:
    """ Split Class of the Streaming Gradient Trees (SGT).

    SGTs only have one type of Node. Their nodes, however, can carry a split object that indicates
    they have been split and are not leaves anymore.
    """
    def __init__(self, feature_idx=None, feature_val=None, is_nominal=False):
        # loss_mean and loss_var are actually statistics of the approximation to the
        # *change* in loss.

        self.loss_mean = 0.0
        self.loss_var = 0.0
        self.delta_pred = None
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.is_nominal = is_nominal


class SGTNode:
    def __init__(self, prediction=0.0, depth=0):
        self._prediction = prediction
        self.depth = depth

        self.reset()

    def reset(self):
        # Split test
        self._split = None
        self._children = None
        self._split_stats = {}
        self._update_stats = GradHessStats()

    def sort_instance(self, x):
        if self._split is None:
            return self

        if self._split.is_nominal:
            try:
                node = self._children[x[self._split.feature_idx]]
            except KeyError:
                # Create new node to encompass the emerging category
                self._children[x[self._split.feature_idx]] = SGTNode(
                    prediction=self._prediction, depth=self.depth + 1)
                node = self._children[x[self._split.feature_idx]]
        else:
            node = self._children[0] if x[self._split.feature_idx] <= self._split.feature_val \
                else self._children[1]

        return node.sort_instance(x)

    def update(self, x, grad_hess, sgt):
        # TODO: deal with missing features (skip update)
        for idx, x_val in x.items():
            if not isinstance(x_val, numbers.Number) or (sgt.nominal_attributes is not None
                                                         and idx in sgt.nominal_attributes):
                try:
                    self._split_stats[idx][x_val].add_instance(grad_hess)
                except KeyError:
                    # categorical features are treated with a simple dict structure
                    self._split_stats[idx] = {}
                    self._split_stats[idx][x] = GradHessStats()
                    self._split_stats[idx][x].add_instance(grad_hess)
            else:
                stats = GradHessStats()
                stats.add_instance(grad_hess, x)
                try:
                    self._split_stats[idx].add(stats)
                except KeyError:
                    # Create a new quantizer
                    quantization_radius = sgt._get_quantization_radius(idx)
                    self._split_stats[idx] = EqualBinFeatureQuantizer(
                        radius=quantization_radius)
                    self._split_stats[idx].add(stats)

        self._update_stats.add_instance(grad_hess)

    def predict(self):
        return self._prediction

    def find_best_split(self, sgt):
        best = SGTSplit()

        # Null split: update the prediction using the new gradient information
        best.delta_pred = self.compute_delta_prediction(self._update_stats.mean(), sgt)
        best.loss_mean = self._update_stats.delta_loss_mean(best.delta_pred)
        best.loss_var = self._update_stats.delta_loss_variance(best.delta_pred)

        for feature_idx in self._split_stats:
            candidate = SGTSplit()
            candidate.feature_idx = feature_idx

            if sgt.nominal_attributes is not None and feature_idx in sgt.nominal_attributes:
                # Nominal attribute has been already used in a previous split
                if feature_idx in sgt._splitted:
                    continue

                candidate.is_nominal = True
                candidate.delta_pred = {}
                loss_mean = 0.0
                loss_var = 0.0
                observations = 0

                cat_collection = self._split_stats[feature_idx]
                for category in cat_collection:
                    p = self.compute_delta_prediction(
                        cat_collection[category].mean(), sgt)

                    m = cat_collection[category].delta_loss_mean(p)
                    s = cat_collection[category].delta_loss_variance(p)

                    n = cat_collection[category].n_observations
                    candidate.delta_pred[category] = p

                    loss_mean = GradHessStats.combine_mean(loss_mean, observations, m, n)
                    loss_var = GradHessStats.combine_variance(loss_mean, loss_var, observations,
                                                              m, s, n)
                    observations += n

                candidate.loss_mean = (loss_mean + len(cat_collection) *
                                       sgt.gamma / self._update_stats.n_observations)
                candidate.loss_var = loss_var
            else:  # Numerical features
                quantizer = self._split_stats[feature_idx]
                n_bins = len(quantizer)

                if n_bins == 1:  # Insufficient bins to perform splits
                    continue

                forward_cumsum = [None for _ in range(n_bins)]
                backward_cumsum = [None for _ in range(n_bins)]

                # Forward cumulative sum
                for j, f in enumerate(quantizer.ordered()):
                    forward_cumsum[j] = quantizer[f].clone()
                    if j > 0:
                        forward_cumsum[j] += forward_cumsum[j - 1]

                # Backward cumulative sum
                for j, b in zip(range(n_bins - 1, -1, -1), quantizer.ordered_rev()):
                    backward_cumsum[j] = quantizer[b].clone()
                    if j + 1 < n_bins:
                        backward_cumsum[j] += backward_cumsum[j + 1]

                candidate.loss_mean = float('Inf')
                candidate.delta_pred = {}

                for j in range(n_bins):
                    delta_pred_left = self.compute_delta_prediction(forward_cumsum[j].mean(), sgt)
                    loss_mean_left = forward_cumsum[j].delta_loss_mean(delta_pred_left)
                    loss_var_left = forward_cumsum[j].delta_loss_variance(delta_pred_left)
                    weight_left = forward_cumsum[j].n_observations

                    delta_pred_right = self.compute_delta_prediction(backward_cumsum[j].mean(),
                                                                     sgt)
                    loss_mean_right = backward_cumsum[j].delta_loss_mean(delta_pred_right)
                    loss_var_right = backward_cumsum[j].delta_loss_variance(delta_pred_right)
                    weight_right = backward_cumsum[j].n_observations

                    loss_mean = GradHessStats.combine_mean(loss_mean_left, weight_left,
                                                           loss_mean_right, weight_right)
                    loss_var = GradHessStats.combine_variance(
                        loss_mean_left, loss_var_left, weight_left,
                        loss_mean_right, loss_var_right, weight_right)

                    if loss_mean < candidate.loss_mean:
                        candidate.loss_mean = (loss_mean + 2.0 * sgt.gamma /
                                               self._update_stats.n_observations)
                        candidate.loss_var = loss_var
                        candidate.delta_pred[0] = delta_pred_left
                        candidate.delta_pred[1] = delta_pred_right

                        # Define split point
                        if j == n_bins - 1:  # Last bin
                            candidate.feature_val = forward_cumsum[j].get_x()
                        else:  # Use middle point between bins
                            candidate.feature_val = (forward_cumsum[j].get_x() +
                                                     forward_cumsum[j + 1].get_x()) / 2.0
            if candidate.loss_mean < best.loss_mean:
                best = candidate

        return best

    def apply_split(self, split, sgt):
        # Null split: update tree prediction and reset learning node
        if split.feature_idx is None:
            self._prediction += split.delta_pred
            sgt._n_node_updates += 1
            self.reset()
            return

        self._split = split
        sgt._n_splits += 1
        sgt._splitted.add(split.feature_idx)

        # Create children
        self._children = {}
        for child_idx, delta_pred in split.delta_pred.items():
            self._children[child_idx] = SGTNode(prediction=self._prediction + delta_pred,
                                                depth=self.depth + 1)
        # Free memory used to monitor splits
        self._split_stats = None
        # Update max depth

        if self.depth + 1 > sgt._max_depth:
            sgt._max_depth = self.depth + 1

    def compute_delta_prediction(self, grad_hess, sgt):
        # Add small constant value to avoid division by zero
        return -grad_hess.gradient / (grad_hess.hessian + sys.float_info.min + sgt.lmbda)

    def is_leaf(self):
        return self._children is None

    @property
    def n_observations(self):
        return self._update_stats.n_observations
