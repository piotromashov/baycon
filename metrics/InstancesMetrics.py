import json
from collections import Counter

import numpy as np
import pandas as pd


def count_and_sort(elements, reverse=False):
    counter = Counter(elements)
    sorted_counter = sorted(counter.items(), key=lambda pair: pair[0], reverse=reverse)
    return sorted_counter


class InstancesMetrics:
    def __init__(self, input_json_filename, distance_calculator):
        with open(input_json_filename) as json_file:
            data = json.load(json_file)
            self._initial_instance = np.array(data["initial_instance"])
            self._counterfactuals = np.array(data["counterfactuals"])
            self._total_time = data["total_time"]
            self._time_to_first_solution = data["time_to_first_solution"] if "time_to_first_solution" in data else None
            self._time_to_best_solution = data["time_to_best_solution"] if "time_to_best_solution" in data else None

        self._similarity_scores = self.calculate_similarity(self._counterfactuals, distance_calculator)
        self._features_changed = self.calculate_features_changed(self._counterfactuals)

    def calculate_similarity(self, counterfactual, distance_calculator):
        return np.around(1 - distance_calculator(self._initial_instance, counterfactual), 4)

    def calculate_features_changed(self, counterfactuals):
        return [sum(counterfactual != self._initial_instance) for counterfactual in counterfactuals]

    def to_csv(self, output_csv_filename):
        df = pd.DataFrame({
            'distance_x': self._similarity_scores,
            'features_changed': self._features_changed,
            'total_time': self._total_time,
            'time_to_first_solution': self._time_to_first_solution,
            'time_to_best_solution': self._time_to_best_solution
        })
        df.to_csv(output_csv_filename)

    def __str__(self):
        metrics_similarities = count_and_sort(self._similarity_scores, reverse=True)
        metrics_features_changed = count_and_sort(self._features_changed)
        return "Distances: {}\nFeatures changed: {}\nTime to first solution: {}\nTotal Time: {}".format(
            str(metrics_similarities),
            str(metrics_features_changed),
            str(self._time_to_first_solution),
            str(self._total_time)
        )
