import json
from collections import Counter

import numpy as np
import pandas as pd

from common.DataAnalyzer import DataAnalyzer
from common.ScoreCalculator import ScoreCalculator
from common.Target import Target


def count_and_sort(elements, reverse=False):
    counter = Counter(elements)
    sorted_counter = sorted(counter.items(), key=lambda pair: pair[0], reverse=reverse)
    return sorted_counter


class InstancesMetrics:
    def __init__(self, dataframe, input_json_filename):
        with open(input_json_filename) as json_file:
            data = json.load(json_file)
            self._initial_instance = np.array(data["initial_instance"])
            self._initial_prediction = np.array(data["initial_prediction"])
            self._target = Target(data["target_type"], data["target_feature"], data["target_value"])
            self._counterfactuals = np.array(data["counterfactuals"])
            self._predictions = np.array(data["predictions"])
            self._total_time = data["total_time"]
            self._time_to_first_solution = data["time_to_first_solution"] if "time_to_first_solution" in data else None
            self._time_to_best_solution = data["time_to_best_solution"] if "time_to_best_solution" in data else None

        data_analyzer = DataAnalyzer(dataframe, self._target)
        self._scores = self.calculate_scores(data_analyzer)
        self._features_changed = self.calculate_features_changed(self._counterfactuals)

    def calculate_scores(self, data_analyzer):
        score_calculator = ScoreCalculator(self._initial_instance, self._initial_prediction, self._target,
                                           data_analyzer)
        return np.around(score_calculator.fitness_score(self._counterfactuals, self._predictions), 4)

    def calculate_features_changed(self, counterfactuals):
        return [sum(counterfactual != self._initial_instance) for counterfactual in counterfactuals]

    def to_csv(self, output_csv_filename):
        df = pd.DataFrame({
            'scores': self._scores,
            'features_changed': self._features_changed,
            'predictions': self._predictions,
            'total_time': self._total_time,
            'time_to_first_solution': self._time_to_first_solution,
            'time_to_best_solution': self._time_to_best_solution
        })
        df.to_csv(output_csv_filename)

    def __str__(self):
        metrics_scores = count_and_sort(self._scores, reverse=True)
        metrics_features_changed = count_and_sort(self._features_changed)
        return "Scores: {}\n" \
               "Features changed: {}\n" \
               "Time to first solution: {}\n" \
               "Total Time: {}".format(
            str(metrics_scores),
            str(metrics_features_changed),
            str(self._time_to_first_solution),
            str(self._total_time)
        )
