import numpy as np

from common.Target import Target


class DataAnalyzer:
    def __init__(self, X, Y, target, cat_features=None, feature_weights=None):
        self._analyze_x(X, cat_features, feature_weights)
        self._analyze_y(Y, target)

    def _analyze_x(self, X, cat_features, feature_weights):
        self._features_count = X.shape[1]
        self._X_min_values = np.min(X, axis=0)
        self._X_max_values = np.max(X, axis=0)

        # check if feature weights info have been provided, if not, infer it
        if np.array(feature_weights).any():
            self._feature_weights = feature_weights
        else:
            self._feature_weights = np.ones(self._features_count)

        # check if categorical features info have been provided, if not, infer it
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
                self._feature_ranges[i] = self._X_max_values[i] - self._X_min_values[i] + 1

    def min_feature_values(self):
        return self._X_min_values

    def max_feature_values(self):
        return self._X_max_values

    def categorical_features(self):
        return self._categorical_features

    def feature_ranges(self):
        return self._feature_ranges

    def feature_weights(self):
        return self._feature_weights

    def _analyze_y(self, Y, target):
        if target.target_type() is not Target.TYPE_CLASSIFICATION:
            self._Y_min_values = np.min(Y, axis=0)
            self._Y_max_values = np.max(Y, axis=0)
        else:
            self._Y_min_values = None
            self._Y_max_values = None

    def prediction_min_value(self):
        return self._Y_min_values

    def prediction_max_value(self):
        return self._Y_max_values
