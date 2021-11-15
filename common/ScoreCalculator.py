import numpy as np

from common.Target import Target

ZERO_VALUE = 0.1


def score_y_away_from_target(min_value, predictions, max_value):
    predictions_diff = np.abs(min_value - predictions)
    total_diff = np.abs(min_value - max_value)
    return (1 - np.divide(predictions_diff, total_diff)) * ZERO_VALUE


def score_y_reaching_target(min_value, predictions, max_value):
    predictions_diff = np.abs(predictions - min_value)
    total_diff = np.abs(max_value - min_value)
    return np.divide(predictions_diff, total_diff)


class ScoreCalculator:
    SCORE_JITTER = 0.75

    def __init__(self, initial_instance, initial_prediction, target, data_analyzer):
        self._initial_instance = initial_instance
        self._initial_prediction = initial_prediction
        self._target = target
        self._data_analyzer = data_analyzer

    def fitness_score(self, instances, predictions):
        # calculate closeness of the potential counterfactual to the initial instance.
        score_x = self.score_x(self._initial_instance, instances)
        score_y = self.score_y(predictions)
        return score_x * score_y

    # returns np.array of gower distances for each instance against the initial one
    def gower_distance(self, origin_instance, instances):
        partial_gowers = np.zeros(instances.shape)
        # repeat for each column (feature values per instances)
        features_count = len(origin_instance)
        for feature_idx in range(features_count):
            target = origin_instance[feature_idx]
            feature_values = instances[:, feature_idx]
            feature_weight = self._data_analyzer.feature_weights()[feature_idx]
            # categorical or numerical, perform calculations accordingly
            if self._data_analyzer.categorical_features()[feature_idx]:
                ij = np.where(feature_values == target, np.zeros_like(feature_values), np.ones_like(feature_values))
            else:
                abs_delta = np.absolute(feature_values - target)
                feature_range = self._data_analyzer.feature_ranges()[feature_idx]
                ij = np.divide(abs_delta, feature_range, out=np.zeros_like(abs_delta), where=feature_range != 0)
            partial_gowers[:, feature_idx] = np.multiply(ij, feature_weight)

        sum_gowers = np.sum(partial_gowers, axis=1)
        gowers = np.divide(sum_gowers, self._data_analyzer.feature_weights().sum())

        return gowers

    def score_x(self, from_instance, to_instances):
        return 1 - self.gower_distance(from_instance, to_instances)

    def score_y(self, predictions_to_calculate):
        if self._target.target_type() == Target.TYPE_CLASSIFICATION:
            return self.calculate_classification_score_y(predictions_to_calculate)
        if self._target.target_type == Target.TYPE_REGRESSION:
            return self.calculate_regression_score_y(predictions_to_calculate)
        if self._target.target_type() == Target.TYPE_RANGE:
            return self.calculate_ranged_score_y(predictions_to_calculate)

    def calculate_classification_score_y(self, predictions):
        return self._target.target_value() == predictions

    def calculate_regression_score_y(self, predictions):
        # calculate for each prediction if is greater or lower than the initial one
        increase_index = self._initial_instance >= predictions
        decrease_index = not increase_index
        max_target_value = self._data_analyzer.prediction_max_value()
        min_target_value = self._data_analyzer.prediction_min_value()

        scores_y = np.array([], dtype=np.float).reshape(predictions.shape[0])
        if self._target.target_value() == Target.REGRESSION_INCREASE:
            scores_y[increase_index] = score_y_reaching_target(
                self._initial_instance, predictions[increase_index], max_target_value)
            scores_y[decrease_index] = score_y_away_from_target(
                self._initial_prediction, predictions[decrease_index], min_target_value)
        elif self._target.target_value() == Target.REGRESSION_DECREASE:
            scores_y[decrease_index] = score_y_reaching_target(
                self._initial_instance, predictions[decrease_index], min_target_value)
            scores_y[increase_index] = score_y_away_from_target(
                self._initial_prediction, predictions[increase_index], max_target_value)
        return scores_y

    def calculate_ranged_score_y(self, predictions):
        start_range, end_range = self._target.target_value()

        # if prediction is in range, then target achieved
        below_range_index = predictions < start_range
        over_range_index = predictions > end_range
        in_range_index = not below_range_index and not over_range_index

        scores_y = np.array([], dtype=np.float).reshape(predictions.shape[0])
        scores_y[in_range_index] = 1

        # if prediction inside the delta slope, assign its score as the current difference divided by total difference
        delta_range_start = np.abs(self._initial_prediction - start_range)
        delta_range_end = np.abs(self._initial_prediction - end_range)
        total_delta_slope = np.min(delta_range_start, delta_range_end)

        left_slope_boundary = start_range - total_delta_slope
        right_slope_boundary = end_range + total_delta_slope
        predictions_within_left_slope_index = left_slope_boundary >= predictions < start_range
        predictions_within_right_slope_index = end_range > predictions <= right_slope_boundary

        scores_y[predictions_within_left_slope_index] = score_y_reaching_target(
            left_slope_boundary, predictions[predictions_within_left_slope_index], start_range)
        scores_y[predictions_within_right_slope_index] = score_y_reaching_target(
            right_slope_boundary, predictions[predictions_within_right_slope_index], end_range)

        # if prediction is outside range and delta slope, then apply penalized score
        outside_left_slope_index = predictions < left_slope_boundary
        outside_right_slope_index = predictions > right_slope_boundary
        min_target_values = self._data_analyzer.prediction_min_value()
        max_target_values = self._data_analyzer.prediction_max_value()

        scores_y[outside_left_slope_index] = score_y_away_from_target(
            left_slope_boundary, predictions[outside_left_slope_index], min_target_values)
        scores_y[outside_right_slope_index] = score_y_away_from_target(
            right_slope_boundary, predictions[outside_right_slope_index], max_target_values)

    def filter_instances_within_score(self, instance_from, instances_to_filter):
        score_x_from_instance = self.score_x(self._initial_instance, np.array([instance_from]))
        scores_to_check = self.score_x(self._initial_instance, instances_to_filter)
        index = self.near_score(score_x_from_instance, scores_to_check)
        return instances_to_filter[index]

    def near_score(self, score, scores_to_check):
        near_scores_index = score >= scores_to_check * self.SCORE_JITTER
        return near_scores_index