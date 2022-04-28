import json
from collections import Counter

import pandas as pd

from common.DataAnalyzer import *
from common.ScoreCalculator import ScoreCalculator
from common.Target import Target


def count_and_sort(elements, reverse=False):
    counter = Counter(elements)
    sorted_counter = sorted(counter.items(), key=lambda pair: pair[0], reverse=reverse)
    return sorted_counter


class InstancesMetrics:
    def __init__(self, dataframe, input_json_filename, model):
        with open(input_json_filename) as json_file:
            data = json.load(json_file)
            target = Target(data["target_type"], data["target_feature"], data["target_value"])
            self._initial_instance = np.array(data["initial_instance"])
            self._initial_prediction = data["initial_prediction"]
            self._target = target
            self._categorical_features = np.array(data["categorical_features"])
            self._actionable_features = np.array(data["actionable_features"])
            self._counterfactuals = np.array(data["counterfactuals"])
            self._predictions = np.array(data["predictions"])
            self._total_time = data["total_time"]
            self._time_to_first_solution = data["time_to_first_solution"] if "time_to_first_solution" in data else None
            self._time_to_best_solution = data["time_to_best_solution"] if "time_to_best_solution" in data else None

            # solve error if prediction is actually a number but was saved as string
            try:
                self._initial_prediction = float(self._initial_prediction)
            except ValueError:
                pass

        Y = dataframe[[target.target_feature()]].values.ravel()
        X = dataframe.drop([target.target_feature()], axis=1).values

        if len(self._categorical_features):
            X = encode(X, self._categorical_features)
        if model == "SVM":
            X = scale(X)

        if len(self._counterfactuals) > 0:
            feature_names = dataframe.columns[dataframe.columns != target.target_feature()]
            data_analyzer = DataAnalyzer(X, Y, feature_names, self._target, self._categorical_features)
            score_calculator = ScoreCalculator(self._initial_instance, self._initial_prediction, self._target,
                                               data_analyzer)
            self._scores, self._scores_x, self._scores_y, self._scores_f = score_calculator.fitness_score(
                self._counterfactuals, self._predictions)
            self._features_changed = self.calculate_features_changed(self._counterfactuals)
        else:
            import warnings
            warnings.warn("Empty counterfactuals for {}".format(input_json_filename))
            self._scores, self._scores_x, self._scores_y, self._scores_f = [], [], [], []
            self._features_changed = []
        self.to_csv(input_json_filename)

    def calculate_features_changed(self, counterfactuals):
        return [sum(counterfactual != self._initial_instance) for counterfactual in counterfactuals]

    def to_csv(self, input_json_filename):
        df = pd.DataFrame({
            'scores': self._scores,
            'scores_x': self._scores_x,
            'scores_y': self._scores_y,
            'scores_f': self._scores_f,
            'features_changed': self._features_changed,
            'predictions': self._predictions,
            'total_time': self._total_time,
            'time_to_first_solution': self._time_to_first_solution,
            'time_to_best_solution': self._time_to_best_solution
        })
        output_csv_filename = input_json_filename.split(".json")[0] + ".csv"
        df.to_csv(output_csv_filename)
        print("--- Finished: saved file {}".format(output_csv_filename))
        return output_csv_filename

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
