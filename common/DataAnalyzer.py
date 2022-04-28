import numpy as np

from common.Target import Target


def encode(X, categorical_features):
    from common.MultiColumnLabelEncoder import MultiColumnLabelEncoder
    return MultiColumnLabelEncoder(categorical_features).fit_transform(X)


def scale(X):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit_transform(X)


class DataAnalyzer:
    def __init__(self, X, Y, feature_names, target, cat_features=None, actionable_features=None, feature_weights=None):
        if feature_weights is None:
            feature_weights = {}
        if cat_features is None:
            cat_features = []
        self._X = X
        self._Y = Y
        self._features_count = self._X.shape[1]
        self._target = target
        self._features = feature_names

        if actionable_features is None or len(actionable_features) == 0:
            actionable_features_mask = [1] * self._features_count
        else:
            actionable_features_mask = [1 if f in actionable_features else 0 for f in self._features]
        self._actionable_features_mask = np.array(actionable_features_mask)

        self._feature_weights = [feature_weights[f] if f in feature_weights else 1 for f in self._features]
        self._categorical_features = np.isin(self._features, cat_features)

        # perform additional check for strings and treat them as categories as well
        for idx in range(len(self._features)):
            try:
                float(self._X[idx][0])
            except ValueError:
                self._categorical_features[idx] = True
        self._numerical_features = np.logical_not(self._categorical_features)
        self._analyze_dataframe()

    def _analyze_dataframe(self):
        self._analyze_x()
        self._analyze_y()

    def _analyze_x(self):
        X_min_values = np.zeros(self._features_count)
        X_max_values = np.zeros(self._features_count)
        X_min_values[self._numerical_features] = np.quantile(self._X[:, self._numerical_features], 0.05, axis=0)
        X_max_values[self._numerical_features] = np.quantile(self._X[:, self._numerical_features], 0.95, axis=0)

        # categorical features shouldn't have minimum and maximum
        self._X_min_values = np.array(
            [None if v else X_min_values[k] for k, v in enumerate(self._categorical_features)]).astype(float)
        self._X_max_values = np.array(
            [None if v else X_max_values[k] for k, v in enumerate(self._categorical_features)]).astype(float)

        # create ranges for features, numerical and categorical
        self._feature_ranges = np.array([None] * self._features_count)
        for idx, is_categorical in enumerate(self._categorical_features):
            if not is_categorical:
                self._feature_ranges[idx] = self._X_max_values[idx] - self._X_min_values[idx] + 1

    def _analyze_y(self):
        if self._target.target_type() is not Target.TYPE_CLASSIFICATION:
            self._Y_min = np.min(self._Y, axis=0)
            self._Y_max = np.max(self._Y, axis=0)
        else:
            self._Y_min = None
            self._Y_max = None

    def data(self):
        return self._X, self._Y

    def min_feature_values(self):
        return self._X_min_values

    def max_feature_values(self):
        return self._X_max_values

    def unique_categorical_values(self):
        return [np.unique(c) for c in self._X[:, self._categorical_features].transpose()]

    def features(self):
        return self._features

    def actionable_features_mask(self):
        return self._actionable_features_mask

    def categorical_features(self):
        return self._categorical_features

    def numerical_features(self):
        return self._numerical_features

    def feature_ranges(self):
        return self._feature_ranges

    def feature_weights(self):
        return self._feature_weights

    def prediction_min_value(self):
        return self._Y_min

    def prediction_max_value(self):
        return self._Y_max
