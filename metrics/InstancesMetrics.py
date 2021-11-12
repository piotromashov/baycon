import json
from collections import Counter

import numpy as np
import pandas as pd

from common.SimilarityCalculator import SimilarityCalculator
from common.Target import Target


def count_and_sort(elements, reverse=False):
    counter = Counter(elements)
    sorted_counter = sorted(counter.items(), key=lambda pair: pair[0], reverse=reverse)
    return sorted_counter


class InstancesMetrics:
    def __init__(self, input_json_filename, data_analyzer):
        with open(input_json_filename) as json_file:
            data = json.load(json_file)
            self._initial_instance = np.array(data["initial_instance"])
            self._initial_prediction = np.array(data["initial_prediction"])
            self._target = Target(data["target_type"], data["target_value"])
            self._counterfactuals = np.array(data["counterfactuals"])
            self._predictions = np.array(data["predictions"])
            self._total_time = data["total_time"]
            self._time_to_first_solution = data["time_to_first_solution"] if "time_to_first_solution" in data else None
            self._time_to_best_solution = data["time_to_best_solution"] if "time_to_best_solution" in data else None

        self._similarity_scores = self.calculate_similarities(data_analyzer)
        self._features_changed = self.calculate_features_changed(self._counterfactuals)

    def calculate_similarities(self, data_analyzer):
        similarity_calculator = SimilarityCalculator(self._initial_instance, self._initial_prediction, self._target,
                                                     data_analyzer)
        return np.around(similarity_calculator.calculate_scores(self._counterfactuals, self._predictions), 4)

    def calculate_features_changed(self, counterfactuals):
        return [sum(counterfactual != self._initial_instance) for counterfactual in counterfactuals]

    def to_csv(self, output_csv_filename):
        df = pd.DataFrame({
            'scores': self._similarity_scores,
            'features_changed': self._features_changed,
            'predictions': self._predictions,
            'total_time': self._total_time,
            'time_to_first_solution': self._time_to_first_solution,
            'time_to_best_solution': self._time_to_best_solution
        })
        df.to_csv(output_csv_filename)

    def __str__(self):
        metrics_similarities = count_and_sort(self._similarity_scores, reverse=True)
        metrics_features_changed = count_and_sort(self._features_changed)
        return "Scores: {}\n" \
               "Features changed: {}\n" \
               "Time to first solution: {}\n" \
               "Total Time: {}".format(
            str(metrics_similarities),
            str(metrics_features_changed),
            str(self._time_to_first_solution),
            str(self._total_time)
        )
