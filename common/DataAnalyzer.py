import numpy as np


# TODO: add analysis of Y field (min_values, max_values)
class DataAnalyzer:
    def __init__(self, X, Y, cat_features=None, feature_weights=None):
        self._features_count = X.shape[1]
        self._min_values = np.min(X, axis=0)
        self._max_values = np.max(X, axis=0)

        if feature_weights:
            self._feature_weights = feature_weights
        else:
            self._feature_weights = np.ones(self._features_count)

        if np.array(cat_features).any():
            self._categorical_features = cat_features
        else:
            self._categorical_features = np.zeros(self._features_count, dtype=bool)
            for col in range(self._features_count):
                if not np.issubdtype(type(X[0, col]), np.number):
                    self._categorical_features[col] = True

        self._numerical_features = np.logical_not(self._categorical_features)

        # create ranges for features, numerical and categorical
        self._feature_ranges = np.array([None] * self._features_count)
        for i in range(len(self._feature_ranges)):
            if not self._categorical_features[i]:
                self._feature_ranges[i] = self._max_values[i] - self._min_values[i] + 1

    def min_feature_values(self):
        return self._min_values

    def max_feature_values(self):
        return self._max_values

    def categorical_features(self):
        return self._categorical_features

    def feature_ranges(self):
        return self._feature_ranges

    def feature_weights(self):
        return self._feature_weights

    def target_min_value(self):
        pass

    def target_max_value(self):
        pass
