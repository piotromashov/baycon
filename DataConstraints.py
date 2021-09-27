import numpy as np

class DataConstraints:
    def __init__(self, dataset):
        self._dataset = dataset
        self._features_count = dataset.shape[1]
        self.generate_minMax_constraints()

    def generate_minMax_constraints(self):
        #TODO: make this dynamically in base of the feature ranges
        self._min_values = np.min(self._dataset, axis=0)
        self._max_values = np.max(self._dataset, axis=0)

    def min_feature_values(self):
        return self._min_values

    def max_feature_values(self):
        return self._max_values