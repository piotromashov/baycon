import numpy as np


class DataConstraints:
    def __init__(self, dataset, cat_features=False):
        self._features_count = dataset.shape[1]
        self._min_values = np.min(dataset, axis=0)
        self._max_values = np.max(dataset, axis=0)

        if cat_features:
            self._categorical_features = cat_features
        else:
            self._categorical_features = np.zeros(self._features_count, dtype=bool)
            for col in range(self._features_count):
                if not np.issubdtype(type(dataset[0, col]), np.number):
                    self._categorical_features[col] = True

        self._numerical_features = np.logical_not(self._categorical_features)

        # create ranges for features, numerical and categorical
        ranges = np.array([0]*self._features_count)
        for i in range(len(ranges)):
            if not self._categorical_features[i]:
                ranges[i] = self._max_values[i] - self._min_values[i] + 1
        self._features_range = ranges

    def min_feature_values(self):
        return self._min_values

    def max_feature_values(self):
        return self._max_values

    def categorical(self):
        return self._categorical_features

    def numerical(self):
        return self._numerical_features

    def feature_weights(self):
        return np.ones(self._features_count)

    def features_count(self):
        return self._features_count

    def features_range(self):
        return self._features_range

    def features_max_distance(self):
        return np.sum((self._max_values - self._min_values) + 1)
