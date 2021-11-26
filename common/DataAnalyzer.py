import numpy as np

from common.MultiColumnLabelEncoder import MultiColumnLabelEncoder
from common.Target import Target


class DataAnalyzer:
    def __init__(self, dataframe, target, cat_features=None, feature_weights=None):
        if feature_weights is None:
            feature_weights = {}
        if cat_features is None:
            cat_features = []
        target_feature = target.target_feature()
        assert target_feature in dataframe.columns
        self._features = dataframe.columns[dataframe.columns != target_feature]
        self._target = target
        self._dataframe = dataframe
        self._feature_weights = [feature_weights[f] if f in feature_weights else 1 for f in self._features]
        self._categorical_features = [True if f in cat_features else False for f in self._features]
        # perform additional check for strings and treat them as categories as well
        for idx, f in enumerate(self._features):
            if not np.issubdtype(type(self._dataframe[f][0]), np.number):
                self._categorical_features[idx] = True
        self._numerical_features = np.logical_not(self._categorical_features)

        self._analyze_dataframe()

    def _analyze_dataframe(self):
        target_feature = self._target.target_feature()
        self._Y = self._dataframe[[target_feature]].values.ravel()
        self._X = self._dataframe.drop([target_feature], axis=1).values
        self._analyze_x()
        self._analyze_y()

    def _analyze_x(self):
        self._features_count = self._X.shape[1]
        X_min_values = np.min(self._X, axis=0)
        X_max_values = np.max(self._X, axis=0)

        # categorical features shouldn't have minimum and maximum
        self._X_min_values = [None if v else X_min_values[k] for k, v in enumerate(self._categorical_features)]
        self._X_max_values = [None if v else X_max_values[k] for k, v in enumerate(self._categorical_features)]

        # create ranges for features, numerical and categorical
        self._feature_ranges = np.array([None] * len(self._features))
        for i in range(len(self._feature_ranges)):
            if not self._categorical_features[i]:
                self._feature_ranges[i] = self._X_max_values[i] - self._X_min_values[i] + 1

    def _analyze_y(self):
        if self._target.target_type() is not Target.TYPE_CLASSIFICATION:
            self._Y_min = np.min(self._Y, axis=0)
            self._Y_max = np.max(self._Y, axis=0)
        else:
            self._Y_min = None
            self._Y_max = None

    def encode(self):
        categorical_feature_names = self._features[self._categorical_features]
        self._mcle = MultiColumnLabelEncoder(columns=categorical_feature_names).fit(self._dataframe)
        self._dataframe = self._mcle.transform(self._dataframe)
        self._analyze_dataframe()

    def decode(self, samples):
        return self._mcle.inverse_transform(samples)

    def data(self):
        return self._X, self._Y

    def min_feature_values(self):
        return self._X_min_values

    def max_feature_values(self):
        return self._X_max_values

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
