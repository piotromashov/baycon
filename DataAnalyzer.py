from DistanceCalculator import *


class DataAnalyzer:
    def __init__(self, dataset, cat_features=False, feature_weights=None):
        self._features_count = dataset.shape[1]
        self._min_values = np.min(dataset, axis=0)
        self._max_values = np.max(dataset, axis=0)

        if feature_weights:
            self._feature_weights = feature_weights
        else:
            self._feature_weights = np.ones(self._features_count)

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

        self._distance_calculator = DistanceCalculator(ranges, self._categorical_features, self._feature_weights)

    def distance_calculator(self):
        return self._distance_calculator

    def min_feature_values(self):
        return self._min_values

    def max_feature_values(self):
        return self._max_values
