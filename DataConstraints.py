import numpy as np


class DataConstraints:
    def __init__(self, dataset):
        self._features_count = dataset.shape[1]
        self._min_values = np.min(dataset, axis=0)
        self._max_values = np.max(dataset, axis=0)

    def min_feature_values(self):
        return self._min_values

    def max_feature_values(self):
        return self._max_values

    def features_max_distance(self):
        return np.sum((self._max_values - self._min_values) + 1)
