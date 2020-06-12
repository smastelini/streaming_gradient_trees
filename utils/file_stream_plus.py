import numpy as np

from skmultiflow.data import FileStream
from skmultiflow.data.data_stream import check_data_consistency
from skmultiflow.utils import check_random_state


class FileStreamPlus(FileStream):
    _MODE_CURRENT = 'current'
    _MODE_FUTURE = 'future'

    """ Modified FileStream class that can handle either numpy.ndarray data or a pandas.DataFrame/
    pandas.Series and dictionaries/scalar combination.

    It also adds support to shuffling the data after reading it from the disk (useful when
    evaluating stationary streams).


    Parameters
    ---------

        # TODO:

        mode: operates in the same way as the Streaming Gradient Trees.
    """
    def __init__(self, filepath, target_idx=-1, n_targets=1, cat_features=None, allow_nan=False,
                 shuffle_data=False, random_state=None, mode='future'):
        self.shuffle_data = shuffle_data
        self.random_state = random_state
        self._random_state = check_random_state(random_state)

        if mode not in [self._MODE_CURRENT, self._MODE_FUTURE]:
            raise ValueError('Invalid mode. Valid values are "current" and "future".')
        self.mode = mode

        super().__init__(filepath, target_idx, n_targets, cat_features, allow_nan)

    def _load_data(self):
        """ Reads the data provided by the user and separates the features and targets.
        """
        try:
            raw_data = self.read_function(self.filepath)
            if self.shuffle_data:
                raw_data = raw_data.sample(frac=1, random_state=self._random_state).\
                    reset_index(drop=True)

            check_data_consistency(raw_data, self.allow_nan)

            rows, cols = raw_data.shape
            self.n_samples = rows
            labels = raw_data.columns.values.tolist()

            if (self.target_idx + self.n_targets) == cols or (
                    self.target_idx + self.n_targets) == 0:
                # Take everything to the right of target_idx
                self.y = raw_data.iloc[:, self.target_idx:]
                self.target_names = raw_data.iloc[:, self.target_idx:].columns.values.tolist()
            else:
                # Take only n_targets columns to the right of target_idx, use the rest as features
                self.y = raw_data.iloc[:, self.target_idx:self.target_idx + self.n_targets]
                self.target_names = labels[self.target_idx:self.target_idx + self.n_targets]

            self.X = raw_data.drop(self.target_names, axis=1)
            self.feature_names = raw_data.drop(self.target_names, axis=1).columns.values.tolist()

            _, self.n_features = self.X.shape
            if self.cat_features_idx:
                if max(self.cat_features_idx) < self.n_features:
                    self.n_cat_features = len(self.cat_features_idx)
                else:
                    raise IndexError('Categorical feature index in {} '
                                     'exceeds n_features {}'.format(self.cat_features_idx,
                                                                    self.n_features))
            self.n_num_features = self.n_features - self.n_cat_features

            if np.issubdtype(self.y.values.dtype, np.integer):
                self.task_type = self._CLASSIFICATION
                self.n_classes = len(np.unique(self.y.values))
            else:
                self.task_type = self._REGRESSION
            self.target_values = self.get_target_values()

            if self.mode == 'current':
                self.X = self.X.values
                self.y = self.y.values
        except FileNotFoundError:
            raise FileNotFoundError("File {} does not exist.".format(self.filepath))

    def next_sample(self, batch_size=1):
        if self.mode == 'current':
            return super().next_sample(batch_size)

        self.sample_idx += batch_size
        try:

            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx]
            self.current_sample_x = self.current_sample_x.reset_index(drop=True)
            self.current_sample_y = self.current_sample_y.reset_index(drop=True)

            if batch_size > 1 and self.n_targets == 1:
                # Get pandas Series
                self.current_sample_y = self.current_sample_y.iloc[:, 0]
            else:
                self.current_sample_x = [self.current_sample_x.to_dict(orient='index')[0]]

                if self.n_targets == 1:
                    self.current_sample_y = [self.current_sample_y.iloc[0][0]]
                else:
                    self.current_sample_y = [self.current_sample_y.to_dict(orient='index')[0]]

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
        return self.current_sample_x, self.current_sample_y

    def get_target_values(self):
        if self.task_type == 'classification':
            if self.n_targets == 1:
                return np.unique(self.y.values).tolist()
            else:
                return [np.unique(self.y.values[:, i]).tolist() for i in range(self.n_targets)]
        elif self.task_type == self._REGRESSION:
            return [float] * self.n_targets
