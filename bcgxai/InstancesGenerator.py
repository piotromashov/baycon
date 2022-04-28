import numpy as np

from common import numpy_utils as npu


class InstancesGenerator:
    INITIAL_NEIGHBOUR_SAMPLING_FACTOR = 200
    NEIGHBOURS_SAMPLING_FACTOR = 100
    ROUNDING = 0.01
    UNIFORM_SAMPLING_FACTOR = 1000

    def __init__(self, initial_instance, data_analyzer, score_calculator):
        self._initial_instance = initial_instance
        self._data_analyzer = data_analyzer
        self._numerical_features = data_analyzer.numerical_features()
        self._min_values = data_analyzer.min_feature_values()
        self._max_values = data_analyzer.max_feature_values()
        self._score_calculator = score_calculator

    def generate_initial_neighbours(self):
        def numerical_sampling_strategy(rows_to_sample):
            numerical_f = self._numerical_features
            means_for_sds = np.mean([self._max_values[numerical_f], self._min_values[numerical_f]], axis=0)
            sds = np.sqrt(np.abs(means_for_sds))  # standard deviation calculated as mean square root
            features_samples = npu.normal_dist_sample(self._initial_instance[numerical_f], sds,
                                                      self._min_values[numerical_f], self._max_values[numerical_f],
                                                      rows_to_sample)
            return self.round_numerical(features_samples, self._initial_instance[numerical_f])

        instances = self.generate(self.INITIAL_NEIGHBOUR_SAMPLING_FACTOR, numerical_sampling_strategy)
        return instances

    def generate_neighbours_arr(self, origin_instances, known_alternatives):
        total_neighbours = np.array([], dtype=np.int64).reshape(0, self._initial_instance.shape[0])
        sampling_factor = self.NEIGHBOURS_SAMPLING_FACTOR//len(origin_instances)
        for origin_instance in origin_instances:
            neighbours = self.generate_neighbours(origin_instance, known_alternatives, sampling_factor)
            total_neighbours = npu.unique_concatenate(total_neighbours, neighbours)
        return total_neighbours

    def generate_neighbours(self, origin_instance, known_alternatives, sampling_factor):
        def numerical_sampling_strategy(rows_to_sample):
            numerical_features = self._numerical_features
            # calculate mean and std based on min and max values
            means = origin_instance[numerical_features]
            sds = np.sqrt(np.abs(means))
            # direction for values to explore for each feature
            increase_index = self._initial_instance >= origin_instance
            decrease_index = self._initial_instance < origin_instance
            tops = np.where(increase_index, self._initial_instance, origin_instance)[numerical_features]
            bottoms = np.where(decrease_index, self._initial_instance, origin_instance)[numerical_features]
            features_samples = npu.normal_dist_sample(means, sds, bottoms, tops, rows_to_sample)
            return self.round_numerical(features_samples, self._initial_instance[self._numerical_features])

        instances = self.generate(sampling_factor, numerical_sampling_strategy)
        instances = self.filter(instances, origin_instance, known_alternatives)
        return instances

    def generate_random(self, best_instance, known_alternatives):
        def numerical_sampling_strategy(rows_to_sample):
            mins, maxs = self._min_values[self._numerical_features], self._max_values[self._numerical_features]
            features_samples = npu.uniform_dist_sample(mins, maxs, rows_to_sample)
            return self.round_numerical(features_samples, self._initial_instance[self._numerical_features])

        instances = self.generate(self.UNIFORM_SAMPLING_FACTOR, numerical_sampling_strategy)
        instances = self.filter(instances, best_instance, known_alternatives)
        return instances

    def generate(self, sampling_factor, numerical_sampling):
        update_features_mask = npu.features_to_update(len(self._initial_instance), sampling_factor)
        rows_to_sample = update_features_mask.shape[0]
        features = np.zeros(shape=(len(self._initial_instance), rows_to_sample))
        features[self._numerical_features] = numerical_sampling(rows_to_sample)
        categorical_features = np.logical_not(self._numerical_features)
        features[categorical_features] = self.categorical_sampling(rows_to_sample)

        generated_instances = features.transpose()
        instances = np.tile(self._initial_instance, (rows_to_sample, 1))
        instances[update_features_mask] = generated_instances[update_features_mask]
        unactionable_mask = np.logical_not(self._data_analyzer.actionable_features_mask())
        instances[:, unactionable_mask] = self._initial_instance[unactionable_mask]

        return instances[np.sum(instances != self._initial_instance, axis=1) > 0]

    def filter(self, instances, from_instance, known_alternatives):
        instances = np.unique(instances, axis=0)
        instances = npu.not_repeated(known_alternatives, instances)
        return self._score_calculator.filter_instances_within_score(from_instance, instances)

    def categorical_sampling(self, sample_size):
        return npu.random_pick(self._data_analyzer.unique_categorical_values(), sample_size)

    def round_numerical(self, features_samples, to_round_values):
        numerical_f = self._numerical_features
        features_step = (np.abs(self._max_values[numerical_f] - self._min_values[numerical_f])) * self.ROUNDING
        step_amounts = np.divide(features_samples.transpose(), features_step,
                                 out=np.zeros_like(features_samples.transpose()),
                                 where=features_step != 0)
        rounded_samples = np.round(step_amounts) * features_step  # round to closest step

        samples_differences_with_initial = np.abs(rounded_samples - to_round_values)
        index_to_round_to_initial = samples_differences_with_initial <= features_step

        for row_index, row in enumerate(np.array(index_to_round_to_initial)):
            rounded_samples[row_index][row] = to_round_values[row]

        return rounded_samples.transpose()
