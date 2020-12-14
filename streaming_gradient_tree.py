import math
import numpy as np
from scipy.stats import f as FTest

from skmultiflow.core import BaseSKMObject, ClassifierMixin, RegressorMixin, MultiOutputMixin
from skmultiflow.utils import get_dimensions
from nodes.sgt_nodes import SGTNode
from objective import BinaryCrossEntropyObjective, SoftmaxCrossEntropyObjective, \
    SquaredErrorObjective


class StreamingGradientTree(BaseSKMObject, ClassifierMixin, RegressorMixin, MultiOutputMixin):
    """ Modified Implementation of the Streaming Gradient Trees_[1].

    This implementation enhances the original proposal by using an incremental strategy to
    discretize numerical features dynamically, rather than relying on a calibration set and
    parameterized number of bins. The strategy used is essentially employing a simplified Locality
    Sensitive Hashing (LSH) model with a single projection (identity function). Different bin size
    setting policies are available for selection. They directly related to number of split
    candidates the tree is going to explore, and thus, how accurate its split decisions are going
    to be. Besides, the number of stored bins per feature is directly related to the tree's memory
    usage and runtime.

    This algorithm operates with a single tree in case of single-target regression tasks. For all
    the other covered learning tasks a comitte of trees is used (one tree for each target or
    class).

    Parameters
    ----------
        delta: float, optional (default=1e-7)
            Define the significance level of the F-tests performed to decide upon creating splits
            or updating predictions.

        grace_period: int, optional (default=200)
            Interval between split attempts or prediction updates.

        init_pred: float, optional (default=0.0)
            Initial value predicted by each individual tree.

        lmbda: float, optional (default=0.1)
            Positive float value used to impose a penalty over the tree's predictions and force
            them become smaller. The greater the lmbda value, the more constrained are the
            predictions.

        gamma: float, optional (default=1)
            Positive float value used to impose a penalty over the tree's split and force them to
            be avoided when possible. The greater the gamma value, the smaller the chance of a
            split occuring.

        nominal_attributes: list, optional (default=None)
            List with names (mode='future') or indices (mode='current') of the nominal attributes.
            If None, all features are assumed to be numeric.

        quantization_strategy: str, optional (default='stddiv3')
            Defines the policy used to define the quantization radius applied on numerical
            attributes' discretization. Each time a new leaf is created, a radius, or
            discretization interval, is assigned to each feature in order to quantize it.
            The smaller the interval used, the more points are going to be stored by the tree
            between splits. While using more points ought to increase the memory footprint and
            runtime, the split candidate potentially are going to be better. When the trees are
            created and no data was observed, the initial radius is defined by the parameter
            'default_radius'. The next nodes are going to use the policy define by this parameter.

            The valid values are:
            * 'stddiv2': use the standard deviation of the feature divided by 2 as the radius.
            * 'stddiv3': use the standard deviation of the feature divided by 3 as the radius.
            * 'stddiv5': use the standard deviation of the feature divided by 5 as the radius.
            * 'stddiv7': use the standard deviation of the feature divided by 7 as the radius.
            * 'stddiv10': use the standard deviation of the feature divided by 10 as the radius.
            * 'constant': use the value defined in 'default_radius' as the static quantization
            interval.

        default_radius: float, optional (default=0.01)
            Quantization radius used when the tree is created and no data is observed or when
            ``quantization_strategy='constant'``.

        classification_threshold: float, optional (default=0.5)
            Threshold used in multi-label classification to binarize the trees outputs in
            {0, 1}. Ultimately, controls the sensitivity of the tree commitee to choose between
            positive and negative labels.

        mode: str, optional (default='future')
            *** ATTENTION: remove in the future ***

            This tricky parameter controls the behavior of the trees regarding the input data.

            The following values are supported:

            * 'future': accepts as input, dictionaries for X and scalars and dictionaries for y, in
            case of single-instance learning. It also accepts pandas.DataFrames for X and
            pandas.Series or pandas.DataFrames for y, in case of mini-batch learning.

            * 'current': operates with numpy.ndarrays as currently done in skmultiflow. Notice that
            this option brings some overhead due to data conversion, as internally the trees use
            dictionaries.

            *** END ATTENTION ***

    Notes
    -----
        The implementation is able to handle single-target and multi-target regression tasks, as
    well as binary, multiclass, and multi-label classification. Multi-target multiclass
    classification is not currently supported and should be handle with aid of the
    ``MultiOutputLearner`` which is available in the ``meta`` submodule.

    References
    ---------
        [1] Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees.
        In Asian Conference on Machine Learning (pp. 1094-1109).
    """

    _STD_DIV_2 = 'stddiv2'
    _STD_DIV_3 = 'stddiv3'
    _STD_DIV_5 = 'stddiv5'
    _STD_DIV_7 = 'stddiv7'
    _STD_DIV_10 = 'stddiv10'
    _CONSTANT_RAD = 'constant'

    _CLASSIFICATION = 'classification'
    _REGRESSION = 'regression'
    _MULTI_OUTPUT = 'multi-output'

    _MODE_CURRENT = 'current'
    _MODE_FUTURE = 'future'

    def __init__(self, delta: float = 1e-7, grace_period: int = 200, init_pred: float = 0.0,
                 lmbda: float = 0.1, gamma: float = 1.0, nominal_attributes: list = None,
                 quantization_strategy: str = 'stddiv3', default_radius: float = 0.01,
                 classification_threshold: float = 0.5, mode='future'):

        self.delta = delta
        self.grace_period = grace_period
        self.init_pred = init_pred

        if lmbda < 0.0:
            raise ValueError('Invalid value: "lmbda" must be positive.')

        if gamma < 0.0:
            raise ValueError('Invalid value: "gamma" must be positive.')

        self.lmbda = lmbda
        self.gamma = gamma
        self.nominal_attributes = nominal_attributes

        if quantization_strategy not in [self._STD_DIV_2, self._STD_DIV_3, self._STD_DIV_5,
                                         self._STD_DIV_7, self._STD_DIV_10, self._CONSTANT_RAD]:
            raise ValueError('Invalid "quantization_strategy": {}'.format(quantization_strategy))
        self.quantization_strategy = quantization_strategy
        self.default_radius = default_radius
        self.classification_threshold = classification_threshold

        if mode not in [self._MODE_CURRENT, self._MODE_FUTURE]:
            raise ValueError('Invalid mode. Valid values are "current" and "future".')
        self.mode = mode

        self.reset()

    def reset(self):
        # It can carry a single tree (single-target regression) or a
        # commitee of trees (binary, multiclass and multi-label classification, and multi-target
        # regression)
        self._roots = {}

        # set used to check whether categorical feature has been already split
        self._splitted = set()
        self._features_mean_var = {}
        self._n_splits = 0
        self._n_node_updates = 0
        self._n_observations = 0
        self._max_depth = 0

        # For automatic detection of task at hand
        self._task_type = None
        self._objective = None
        self._is_multi_output = False
        self._n_targets = 1

        # Apply different transformations of input/output data before using them to train or to
        # predict: tricky
        self._target_transformer = None
        self._inverse_target_transformer = None

        # Call Different methods depending on the input data type: also very tricky
        if self.mode == self._MODE_CURRENT:
            self._partial_fit_method = self._partial_fit_current
            self._predict_method = self._predict_current
        else:
            # Works with pandas DataFrames and dictionaries
            self._partial_fit_method = self._partial_fit_future
            self._predict_method = self._predict_future

    def _update_global_stats(self, X):
        """ Update inner tree statistics with a new observation.

            Use Welford's algorithm to keep incremental estimates of features' mean and
            variance.
        """
        self._n_observations += 1

        # TODO: are tuples the best approach to store the mean and variance?

        # To avoid checks inside the loop: create separate loops
        if self.nominal_attributes is None:
            for feature_idx, x in X.items():
                try:
                    M, S = self._features_mean_var[feature_idx]
                except KeyError:
                    M, S = 0.0, 0.0

                old_M = M
                M += (x - M) / self._n_observations
                S += (x - M) * (x - old_M)

                self._features_mean_var[feature_idx] = (M, S)
        else:
            for idx, x in X.items():
                for feature_idx, x in enumerate(X):
                    # Skip nominal attributes
                    if feature_idx in self.nominal_attributes:
                        continue
                    try:
                        M, S = self._features_mean_var[feature_idx]
                    except KeyError:
                        M, S = 0.0, 0.0

                    old_M = M
                    M += (x - M) / self._n_observations
                    S += (x - M) * (x - old_M)

                    self._features_mean_var[feature_idx] = (M, S)

    def _get_quantization_radius(self, feature_idx):
        """ Get the quantization radius for a given input feature. """
        if self.quantization_strategy == self._CONSTANT_RAD:
            return self.default_radius

        if self._n_observations < 2:
            # Default radius for quantization: might create too many bins at first
            return self.default_radius

        _, S = self._features_mean_var[feature_idx]
        std_ = math.sqrt(S / (self._n_observations - 1))

        if math.isclose(std_, 0.0):
            return self.default_radius

        if self.quantization_strategy == self._STD_DIV_3:
            return std_ / 3.0
        elif self.quantization_strategy == self._STD_DIV_2:
            return std_ / 2.0
        elif self.quantization_strategy == self._STD_DIV_5:
            return std_ / 5.0
        elif self.quantization_strategy == self._STD_DIV_7:
            return std_ / 7.0
        elif self.quantization_strategy == self._STD_DIV_10:
            return std_ / 10.0
        else:
            return self.default_radius  # Default radius for quantization

    def _identity_helper(self, data):
        """ Helper method that simple returns what it receives as input.

        Used in learning tasks that do not require target transformation before training or after
        predicting.
        """
        return data

    def _transform_target_classification(self, y):
        """ Use a One-Hot enconding strategy to handle multiclass problems. """

        # TODO: verify semi-supervised option (available in MOA)

        # Use comittee of binary trees to deal with multiclass problems
        label = {class_idx: 0 for class_idx in self._roots}
        label[y] = 1

        return label

    def _inverse_transform_target_classification(self, preds):
        """ Return the class whose probability is the highest one. """
        return max(preds, key=preds.get)

    def _inverse_transform_target_multi_label(self, preds):
        """ Binarize the outputs of a tree commitee dealing with multi-label classification. """

        # TODO: should <= be used instead?

        return {target_idx: 0 if val < self.classification_threshold else 1
                for target_idx, val in preds.items()}

    def _transform_target_regression(self, y):
        """ Transform scalar into a dictionary with a single element.

        This action is performed to ensure compability with the other learning tasks.
        """
        return {0: y}

    def _inverse_transform_target_regression(self, preds):
        """ Retrieves the single element stored in the dictionary representation returned
        by the tree in case of a single-target regression task. """
        return preds[0]

    def _init_trees(self, y):
        """ Define the loss function, learning task, and target transformation methods depending on
        the type of the target(s).

        It also initializes the minimum number of needed trees for each learning task. In all
        tasks except classification this number is static. Classification tasks, on the other hand,
        might enconter new incoming classes, which are handled dynamically by creating new trees
        in the comitte.
        """

        # TODO: enable choosing manually the type of learning task?

        if hasattr(y, '__len__') and len(y) > 1:
            self._is_multi_output = True
            self._n_targets = len(y)
            aux_target = y[list(y.keys())[0]]
        else:
            aux_target = y

        if isinstance(aux_target, int) or isinstance(aux_target, np.integer):  # Classification
            self._task_type = self._CLASSIFICATION
            if self._is_multi_output:
                self._objective = BinaryCrossEntropyObjective()
                self._roots = {class_idx: SGTNode(prediction=self.init_pred)
                               for class_idx in range(self._n_targets)}
                # Does not require target transformation
                self._target_transformer = self._identity_helper
                self._inverse_target_transformer = self._inverse_transform_target_multi_label
            else:  # Binary or multiclass classification
                self._objective = SoftmaxCrossEntropyObjective()
                self._roots = {y: SGTNode(prediction=self.init_pred)}
                self._target_transformer = self._transform_target_classification
                self._inverse_target_transformer = self._inverse_transform_target_classification
        else:  # Regression
            self._task_type = self._REGRESSION
            self._objective = SquaredErrorObjective()

            if self._is_multi_output:
                self._roots = {target_idx: SGTNode(prediction=self.init_pred)
                               for target_idx in range(self._n_targets)}
                # Does not require target transformation nor inverse transformation
                self._target_transformer = self._identity_helper
                self._inverse_target_transformer = self._identity_helper
            else:  # Single-target regression
                self._roots = {0: SGTNode(prediction=self.init_pred)}
                self._target_transformer = self._transform_target_regression
                self._inverse_target_transformer = self._inverse_transform_target_regression

    def _update_tree(self, X, grad_hess, target_idx):
        """ Update Streaming Gradient Tree with a single instance. """
        if target_idx not in self._roots:  # Emerging class
            self._roots[target_idx] = SGTNode(prediction=self.init_pred)

        leaf = self._roots[target_idx].sort_instance(X)
        leaf.update(X, grad_hess, self)

        if leaf.total_weight % self.grace_period != 0:
            return

        best_split = leaf.find_best_split(self)

        p = StreamingGradientTree._compute_p_value(best_split, leaf.total_weight)
        if p < self.delta and best_split.loss_mean < 0.0:
            leaf.apply_split(best_split, self)

    @staticmethod
    def _compute_p_value(split, n_observations):
        # Null hypothesis: expected loss is zero
        # Alternative hypothesis: expected loss is not zero

        F = n_observations * (split.loss_mean * split.loss_mean) / split.loss_var \
            if split.loss_var > 0.0 else None

        if F is None:
            return 1.0

        return 1 - FTest.cdf(F, 1, n_observations - 1)

    def fit_one(self, X, y):
        """ Core learning method.

        Learn from a single instance.

        Parameters
        ----------
            X: dict
                Dictionary where the keys identity the features and the values represent the
                observed values.
            y: dict
                Dictionary containing the target(s) identifier(s) and corresponding value(s).
        """
        self._update_global_stats(X)

        if self._task_type is None:
            self._init_trees(y)

        preds = self._predict_one_raw(X)
        label = self._target_transformer(y)

        grad_hess = self._objective.compute_derivatives(label, preds)

        # Update trees with gradient/hessian data
        for target_idx in grad_hess:
            self._update_tree(X, grad_hess[target_idx], target_idx)

    def _predict_one_raw(self, X):
        """ Obtain raw predictions for a single instance. """
        if self._task_type is None:
            return None  # Model was not initialized yet

        preds = {target_class_idx: tree.sort_instance(X).leaf_prediction()
                 for target_class_idx, tree in self._roots.items()}
        return self._objective.transfer(preds)

    def predict_one(self, X):
        """ Core predict method.

        Operates with a single instance at time. Applies the inverse target transformation before
        returning the predictions.

        Parameters
        ----------
            X: dict
                Dictionary containing the feature identifiers (keys) and their values.
        """
        preds = self._predict_one_raw(X)
        return self._inverse_target_transformer(preds)

    def predic_proba_one(self, X):
        """ Core predict_proba method.

        Operates with a single instance at time.

        Parameters
        ----------
            X: dict
                Dictionary containing the feature identifiers (kJust ieys) and their values.
        """
        if self._task_type is None:
            return None  # Model was not initialized yet
        elif self._task_type == self._REGRESSION:
            raise NotImplementedError('predict_proba only applies to classification problems.')

        preds = {target_class_idx: tree.sort_instance(X).leaf_prediction()
                 for target_class_idx, tree in self._roots.items()}
        preds = self._objective.transfer(preds)

        return preds

    # The {fit/predict}_one methods work with dictionaries
    # There are going to exist some repeated logic in partial_fit and predict to offer backward
    # compability with skmultiflow < 0.6
    def _partial_fit_current(self, X, y, sample_weight=None, classes=None):
        """ Main training access point when using numpy.ndarrays.

        Perform needed convertions to dictionaries before calling fit_one.

        Parameters
        ----------
            X: numpy.ndarray or numpy.array
            y: numpy.ndarray or numpy.array
            sample_weight: list-like or numpy.array, optional (default=None)
            classes: not used
        """
        n_rows, _ = get_dimensions(X)
        if sample_weight is None:
            sample_weight = np.ones(n_rows)
        else:
            if len(sample_weight) != n_rows:
                raise ValueError('The weights do not match the number of samples.')

        # Some repeated logic due to backward compability
        if self._task_type is None:
            try:
                if n_rows > 1:
                    self._init_trees({target_idx: val for target_idx, val in enumerate(y[0])})
                else:
                    self._init_trees({target_idx: val for target_idx, val in enumerate(y)})
            except TypeError:
                if n_rows > 1:
                    self._init_trees(y[0])
                else:
                    self._init_trees(y)

        if n_rows == 1 and len(X.shape) == 1:
            X, y = [X], [y]

        for i in range(n_rows):
            x_arr, y_, w = X[i], y[i], sample_weight[i]

            # Internally the trees work with dictionaries
            if self._is_multi_output:
                y_ = {target_idx: val for target_idx, val in enumerate(y_)}
            x = {feat_idx: val for feat_idx, val in enumerate(x_arr)}
            for _ in range(int(w)):
                self.fit_one(x, y_)

    def _partial_fit_future(self, X, y, sample_weight=None, classes=None):
        """ Main training access point when using pandas.DataFrames and dictionaries.

        Parameters
        ----------
            X: dictionary or pandas.DataFrame
            y: scalar, dictionary or pandas.Series
            sample_weight: list-like
            classes: not used
        """
        if isinstance(X, dict):
            n_rows = 1
        else:
            n_rows = len(X)
            if n_rows > 1:
                X = X.to_dict(orient='index')
                X = [X[i] for i in sorted(X)]

                try:
                    y = y.to_dict(orient='index')
                    y = [y[i] for i in sorted(y)]
                except TypeError:
                    y = y.values.tolist()
            else:
                X, y = X[0], y[0]

        if sample_weight is None:
            sample_weight = np.ones(n_rows)
        else:
            if len(sample_weight) != n_rows:
                raise ValueError('The weights do not match the number of samples.')

        if n_rows == 1:
            X, y = [X], [y]

        # Some repeated logic due to backward compability
        if self._task_type is None:
            self._init_trees(y[0])

        for i in range(n_rows):
            x, y_, w = X[i], y[i], sample_weight[i]

            for _ in range(int(w)):
                self.fit_one(x, y_)

    def _predict_current(self, X):
        """ Main predict method when using numpy.ndarrays. """
        if self._task_type is None:
            return None

        dtype = np.int if self._task_type == self._CLASSIFICATION else np.float
        n_rows, _ = get_dimensions(X)
        predictions = np.zeros((n_rows, self._n_targets), dtype=dtype) if self._is_multi_output \
            else np.zeros((n_rows), dtype=dtype)

        if n_rows == 1 and len(X.shape) == 1:
            X = [X]

        for i in range(n_rows):
            x = {feat_idx: val for feat_idx, val in enumerate(X[i])}
            pred = self.predict_one(x)
            predictions[i] = pred if not self._is_multi_output else \
                np.asarray([pred[k] for k in sorted(pred)])

        return predictions

    def _predict_future(self, X):
        """ Main predict method when using pandas.DataFrames or dictionaries. """
        if self._task_type is None:
            return None

        dtype = np.int if self._task_type == self._CLASSIFICATION else np.float
        if isinstance(X, dict):
            n_rows = 1
        else:
            n_rows = len(X)
            if n_rows > 1:
                X = X.to_dict(orient='index')
                X = [X[i] for i in sorted(X)]
            else:
                X = X[0]

        predictions = np.zeros((n_rows, self._n_targets), dtype=dtype) if self._is_multi_output \
            else np.zeros((n_rows), dtype=dtype)

        if n_rows == 1:
            X = [X]

        for i in range(n_rows):
            pred = self.predict_one(X[i])
            predictions[i] = pred if not self._is_multi_output else \
                np.asarray([pred[k] for k in sorted(pred)])

        return predictions

    def partial_fit(self, X, y, sample_weight=None, classes=None):
        """ Call appropriate partial_fit method depending on the type of inputs. """
        self._partial_fit_method(X, y, sample_weight, classes)

    def predict(self, X):
        """ Call the appropriate prediction method depending on the type of inputs.

        Notice that numpy.ndarray are returned regardless of the input type. This action is
        intended to enable compatibility with EvaluatePrequential and other skmultiflow structures.
        The output data type might change in the future.

        *** ATTENTION: remove in the future
            Currently, these input/output interfaces are types of Frankeinstein. They receive one
            type of data but not necessarily return the same type. Intended for testing purposes
            only.
        *** END ATTENTION ***
        """
        return self._predict_method(X)

    def predict_proba(self, X):
        # TODO
        pass

    @property
    def n_nodes(self):
        if len(self._roots) == 0:
            return 0

        n = 0
        for root in self._roots.values():
            to_visit = [root]
            while len(to_visit) > 0:
                node = to_visit.pop(0)
                n += 1

                if node._children is not None:
                    for child in node._children.values():
                        to_visit.append(child)
        return n

    @property
    def n_splits(self):
        return self._n_splits

    @property
    def n_node_updates(self):
        return self._n_node_updates

    @property
    def n_observations(self):
        return self._n_observations

    @property
    def max_depth(self):
        return self._max_depth
