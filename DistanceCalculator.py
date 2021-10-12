import numpy as np


class DistanceCalculator:
    def __init__(self, ranges, categorical, weights):
        self._features_range = ranges
        self._features_categorical = categorical
        self._weights = weights

    # returns np.array of gower distances for each instance against the initial one
    def gower(self, origin_instance, instances):
        partial_gowers = np.zeros(instances.shape)
        # repeat for each column (feature values per instances)
        features_count = len(origin_instance)
        for col_idx in range(features_count):
            target = origin_instance[col_idx]
            feature_values = instances[:, col_idx]
            feature_weight = self._weights[col_idx]
            # categorical or numerical, perform calculations accordingly
            if self._features_categorical[col_idx]:
                ij = np.where(feature_values == target, np.zeros_like(feature_values), np.ones_like(feature_values))
            else:
                abs_delta = np.absolute(feature_values - target)
                feature_range = self._features_range[col_idx]
                ij = np.divide(abs_delta, feature_range, out=np.zeros_like(abs_delta), where=feature_range != 0)
            partial_gowers[:, col_idx] = np.multiply(ij, feature_weight)

        sum_gowers = np.sum(partial_gowers, axis=1)
        gowers = np.divide(sum_gowers, self._weights.sum())

        return gowers
