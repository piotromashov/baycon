import numpy as np

from common import numpy_utils as npu


class InstancesGenerator:
    NEIGHBOURS_SAMPLE_SIZE = 100
    UNIFORM_SAMPLE_SIZE = 1000

    def __init__(self, template, data_analyzer, score_calculator):
        self._initial_instance = template
        self._data_analyzer = data_analyzer
        self._numerical_features = data_analyzer.numerical_features()
        self._min_values = data_analyzer.min_feature_values()
        self._max_values = data_analyzer.max_feature_values()
        self._features = data_analyzer.features()
        self._score_calculator = score_calculator

    def generate_random(self, best_instance, known_alternatives):
        instances = npu.uniform_dist_sample(self._min_values, self._max_values, self.UNIFORM_SAMPLE_SIZE)
        # remove samples that are same as the template
        instances = instances[np.sum(instances != self._initial_instance, axis=1) > 0]
        instances = self._score_calculator.filter_instances_within_score(best_instance, instances)
        instances = npu.not_repeated(known_alternatives, instances)
        return instances

    def generate_initial_neighbours(self):
        # obtain unique labels for each feature
        features = np.zeros(shape=(len(self._numerical_features), self.NEIGHBOURS_SAMPLE_SIZE))

        # for each numerical feature: generate normal distribution and get samples
        means = np.mean([self._max_values[self._numerical_features], self._min_values[self._numerical_features]],
                        axis=0)
        features[self._numerical_features] = npu.normal_dist_sample(
            self._initial_instance[self._numerical_features],
            np.sqrt(np.abs(means.astype(float))),  # standard deviation calculated as mean square root
            self._min_values[self._numerical_features],
            self._max_values[self._numerical_features],
            self.NEIGHBOURS_SAMPLE_SIZE
        )

        # for each categorical feature: pick one randomly from its categories/labels
        features[np.logical_not(self._numerical_features)] = npu.random_pick(
            self._data_analyzer.unique_categorical_values(),
            self.NEIGHBOURS_SAMPLE_SIZE
        )

        return features.transpose()

    def generate_neighbours_arr(self, origin_instances, known_alternatives):
        total_neighbours = np.array([], dtype=np.int64).reshape(0, self._initial_instance.shape[0])
        for origin_instance in origin_instances:
            neighbours = self.generate_neighbours(origin_instance, known_alternatives)
            total_neighbours = npu.unique_concatenate(total_neighbours, neighbours)
        return total_neighbours

    def generate_neighbours(self, origin_instance, known_alternatives):
        features = np.zeros(shape=(len(self._numerical_features), self.NEIGHBOURS_SAMPLE_SIZE))
        # calculate mean and std based on min and max values
        means = origin_instance
        sds = np.sqrt(np.abs(means).astype(float))
        # direction for values to explore for each feature
        increase_index = self._initial_instance >= origin_instance
        decrease_index = self._initial_instance < origin_instance
        tops = np.where(increase_index, self._initial_instance, origin_instance)
        bottoms = np.where(decrease_index, self._initial_instance, origin_instance)

        features[self._numerical_features] = npu.normal_dist_sample(
            means[self._numerical_features],
            sds[self._numerical_features],
            bottoms[self._numerical_features],
            tops[self._numerical_features],
            self.NEIGHBOURS_SAMPLE_SIZE
        )
        # for each categorical feature: pick one randomly from its categories/labels
        features[np.logical_not(self._numerical_features)] = npu.random_pick(
            self._data_analyzer.unique_categorical_values(),
            self.NEIGHBOURS_SAMPLE_SIZE
        )
        neighbours = features.transpose()

        # filter all instances with less score than current instance
        neighbour_scores_x = self._score_calculator.score_x(origin_instance, neighbours)
        score_x_from_origin_to_initial = self._score_calculator.score_x(self._initial_instance,
                                                                        np.array([origin_instance]))
        neighbours = neighbours[neighbour_scores_x >= score_x_from_origin_to_initial]
        unique_neighbours = npu.not_repeated(known_alternatives, neighbours)
        return unique_neighbours
