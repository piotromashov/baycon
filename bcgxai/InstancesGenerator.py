import numpy as np

from common import numpy_utils as npu


class InstancesGenerator:
    INITIAL_NEIGHBOUR_SAMPLE_SIZE = 1000
    NEIGHBOURS_SAMPLE_SIZE = 100
    UNIFORM_SAMPLE_SIZE = 1000
    ROUNDING = 0.01

    def __init__(self, initial_instance, data_analyzer, score_calculator):
        self._initial_instance = initial_instance
        self._data_analyzer = data_analyzer
        self._numerical_features = data_analyzer.numerical_features()
        self._min_values = data_analyzer.min_feature_values()
        self._max_values = data_analyzer.max_feature_values()
        self._score_calculator = score_calculator

    def generate_random(self, best_instance, known_alternatives):
        features = np.zeros(shape=(len(self._numerical_features), self.UNIFORM_SAMPLE_SIZE))
        features[self._numerical_features] = self.rounded_numerical_samples_uniform(self.UNIFORM_SAMPLE_SIZE)
        categorical_features = np.logical_not(self._numerical_features)
        features[categorical_features] = self.categorical_samples_uniform(self.UNIFORM_SAMPLE_SIZE)

        instances = features.transpose()
        instances = instances[np.sum(instances != self._initial_instance, axis=1) > 0]
        instances = self._score_calculator.filter_instances_within_score(best_instance, instances)
        instances = npu.not_repeated(known_alternatives, instances)
        return instances

    def generate_initial_neighbours(self):
        features = np.zeros(shape=(len(self._initial_instance), self.INITIAL_NEIGHBOUR_SAMPLE_SIZE))
        means = np.mean([self._max_values[self._numerical_features], self._min_values[self._numerical_features]],
                        axis=0)
        sds = np.sqrt(np.abs(means.astype(float)))  # standard deviation calculated as mean square root
        features[self._numerical_features] = self.rounded_numerical_samples_normal(means, sds,
                                                                                   self._min_values,
                                                                                   self._max_values,
                                                                                   self.INITIAL_NEIGHBOUR_SAMPLE_SIZE)
        categorical_features = np.logical_not(self._numerical_features)
        features[categorical_features] = self.categorical_samples_uniform(self.INITIAL_NEIGHBOUR_SAMPLE_SIZE)
        instances = features.transpose()
        return instances[np.sum(instances != self._initial_instance, axis=1) > 0]

    def generate_neighbours_arr(self, origin_instances, known_alternatives):
        total_neighbours = np.array([], dtype=np.int64).reshape(0, self._initial_instance.shape[0])
        for origin_instance in origin_instances:
            neighbours = self.generate_neighbours(origin_instance, known_alternatives)
            total_neighbours = npu.unique_concatenate(total_neighbours, neighbours)
        return total_neighbours[np.sum(total_neighbours != self._initial_instance, axis=1) > 0]

    def generate_neighbours(self, origin_instance, known_alternatives):
        features = np.zeros(shape=(len(origin_instance), self.NEIGHBOURS_SAMPLE_SIZE))
        # calculate mean and std based on min and max values
        means = origin_instance
        sds = np.sqrt(np.abs(means).astype(float))
        # direction for values to explore for each feature
        increase_index = self._initial_instance >= origin_instance
        decrease_index = self._initial_instance < origin_instance
        tops = np.where(increase_index, self._initial_instance, origin_instance)
        bottoms = np.where(decrease_index, self._initial_instance, origin_instance)

        features[self._numerical_features] = self.rounded_numerical_samples_normal(means, sds, bottoms, tops,
                                                                                   self.NEIGHBOURS_SAMPLE_SIZE)
        # for each categorical feature: pick one randomly from its categories/labels
        categorical_features = np.logical_not(self._numerical_features)
        features[categorical_features] = self.categorical_samples_uniform(self.NEIGHBOURS_SAMPLE_SIZE)
        neighbours = features.transpose()

        # filter all instances with less score than current instance
        neighbour_scores_x = self._score_calculator.score_x(origin_instance, neighbours)
        score_x_from_origin_to_initial = self._score_calculator.score_x(self._initial_instance,
                                                                        np.array([origin_instance]))
        neighbours = neighbours[neighbour_scores_x >= score_x_from_origin_to_initial]
        unique_neighbours = npu.not_repeated(known_alternatives, neighbours)
        return unique_neighbours

    # get diff for each sample, if it is less than a delta %, then assign value of initial instance there
    def rounded_numerical_samples_normal(self, means, sds, bottoms, tops, sample_size):
        numerical_f = self._numerical_features
        features_samples = npu.normal_dist_sample(means[numerical_f], sds[numerical_f], bottoms[numerical_f],
                                                  tops[numerical_f], sample_size)
        return self.round_numerical(features_samples, self._initial_instance[numerical_f])

    def rounded_numerical_samples_uniform(self, sample_size):
        features_samples = npu.uniform_dist_sample(self._min_values[self._numerical_features],
                                                   self._max_values[self._numerical_features],
                                                   sample_size)
        return self.round_numerical(features_samples, self._initial_instance[self._numerical_features])

    # for each categorical feature: pick one randomly from its categories/labels
    def categorical_samples_uniform(self, sample_size):
        return npu.random_pick(self._data_analyzer.unique_categorical_values(), sample_size)

    def round_numerical(self, features_samples, to_round_values):
        numerical_f = self._numerical_features
        features_step = (np.abs(self._max_values[numerical_f] - self._min_values[numerical_f])) * self.ROUNDING
        step_amounts = np.divide(features_samples.transpose(), features_step,
                                 out=np.zeros_like(features_samples.transpose()),
                                 where=features_step != 0)
        rounded_samples = np.round(step_amounts) * features_step  # round to closest step

        samples_differences_with_initial = np.abs(rounded_samples - to_round_values)
        index_to_round_to_initial = samples_differences_with_initial < features_step

        for row_index, row in enumerate(index_to_round_to_initial):
            rounded_samples[row_index][row] = to_round_values[row]

        return rounded_samples.transpose()
