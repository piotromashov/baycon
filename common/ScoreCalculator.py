import numpy as np


class ScoreCalculator:
    SCORE_JITTER = 0.75

    def __init__(self, initial_instance, initial_prediction, target, data_analyzer):
        self._initial_instance = initial_instance
        self._initial_prediction = initial_prediction
        self._target = target
        self._features_range = data_analyzer.feature_ranges()
        self._features_categorical = data_analyzer.categorical_features()
        self._weights = data_analyzer.feature_weights()

    def fitness_score(self, instances, predictions):
        # calculate closeness of the potential counterfactual to the initial instance.
        score_x = self.score_x(self._initial_instance, instances)
        score_y = self.score_y(self._initial_prediction, predictions)
        return score_x * score_y

    # returns np.array of gower distances for each instance against the initial one
    def gower_distance(self, origin_instance, instances):
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

    def score_x(self, from_instance, to_instances):
        return 1 - self.gower_distance(from_instance, to_instances)

    def score_y(self, from_prediction, to_predictions):
        # TODO: calculate score on Y (based on type of target)
        return self._target.target_value() == to_predictions

    def filter_instances_within_score(self, instance_from, instances_to_filter):
        score_x_from_instance = self.score_x(self._initial_instance, np.array([instance_from]))
        scores_to_check = self.score_x(self._initial_instance, instances_to_filter)
        index = self.near_score(score_x_from_instance, scores_to_check)
        return instances_to_filter[index]

    def near_score(self, score, scores_to_check):
        near_scores_index = score >= scores_to_check * self.SCORE_JITTER
        return near_scores_index
