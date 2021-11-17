import numpy as np

from common.Target import Target


class DataAnalyzer:
    def __init__(self, dataframe, target, cat_features=None, feature_weights=None):
        if feature_weights is None:
            feature_weights = {}
        if cat_features is None:
            cat_features = []
        target_feature = target.target_feature()
        assert target_feature in dataframe.columns
        self._Y = dataframe[[target_feature]].values.ravel()
        dataframe.drop([target_feature], axis=1, inplace=True)
        self._X = dataframe.values

        self._analyze_x(dataframe.columns, cat_features, feature_weights)
        self._analyze_y(target)

    def split_dataset(self):
        return self._X, self._Y

    def _analyze_x(self, feature_names, cat_features, feature_weights):
        self._features_count = self._X.shape[1]
        self._X_min_values = np.min(self._X, axis=0)
        self._X_max_values = np.max(self._X, axis=0)

        self._feature_weights = [feature_weights[f] if f in feature_weights else 1 for f in feature_names]
        self._categorical_features = [True if f in cat_features else False for f in feature_names]
        # perform additional check for strings and treat them as categories as well
        for idx in range(len(feature_names)):
            if not np.issubdtype(type(self._X[0, idx]), np.number):
                self._categorical_features[idx] = True
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

    def _analyze_y(self, target):
        if target.target_type() is not Target.TYPE_CLASSIFICATION:
            self._Y_min = np.min(self._Y, axis=0)
            self._Y_max = np.max(self._Y, axis=0)
        else:
            self._Y_min = None
            self._Y_max = None

    def prediction_min_value(self):
        return self._Y_min

    def prediction_max_value(self):
        return self._Y_max
