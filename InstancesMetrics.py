from collections import Counter

import numpy as np


def count_and_sort(elements, reverse=False):
    counter = Counter(elements)
    sorted_counter = sorted(counter.items(), key=lambda pair: pair[0], reverse=reverse)
    return sorted_counter


class InstancesMetrics:
    def __init__(self, initial_instance, counterfactuals, distance_calculator):
        self._initial_instance = initial_instance
        self._similarity_scores = self.calculate_similarity(counterfactuals, distance_calculator)
        self._features_changed = self.calculate_features_changed(counterfactuals)

    def calculate_similarity(self, counterfactual, distance_calculator):
        return np.around(1 - distance_calculator(self._initial_instance, counterfactual), 3)

    def calculate_features_changed(self, counterfactuals):
        return [sum(counterfactual != self._initial_instance) for counterfactual in counterfactuals]

    def __str__(self):
        metrics_similarities = count_and_sort(self._similarity_scores, reverse=True)
        metrics_features_changed = count_and_sort(self._features_changed)
        return str(metrics_similarities) + "\n" + str(metrics_features_changed)
