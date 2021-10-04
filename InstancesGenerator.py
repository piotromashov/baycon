from numpy.random import default_rng
import numpy_utils as npu
import numpy as np
from sklearn.utils.extmath import cartesian

RANDOM_SAMPLE_SIZE = 10000
CHANGES_JITTER = 8          # TODO: put 20% of feature size
NEIGHBOUR_MAX_DEGREE = 3


class InstancesGenerator:
    def __init__(self, template, data_constraints):
        self._template = template
        self._min_values = data_constraints.min_feature_values()
        self._max_values = data_constraints.max_feature_values()

    def generate_random(self, distance, known_alternatives):
        num_changes = distance * CHANGES_JITTER

        features_amount = len(self._template)
        if num_changes > features_amount / 2:
            num_changes = num_changes // 2

        min_change = 1
        if not num_changes:
            num_changes = min_change

        # maximum and minimum allowed change for one attribute (e.g., from 0 to 3)
        max_value = max(self._max_values)
        min_value = -max_value

        random_matrix = np.random.randint(min_value, max_value, (RANDOM_SAMPLE_SIZE, features_amount))

        instances = random_matrix + self._template  # increase each row with the template (i.e., increase template values by zero,one or two)

        # remove values bigger than max value
        while len(instances[instances > self._max_values]) > 0:
            instances[instances > self._max_values] = instances[instances > self._max_values] - 1
        instances[instances < self._min_values] = 0

        instances = instances.astype(int)
        # remove samples that are same as the template
        instances = instances[np.sum(instances != self._template, axis=1) > 0]

        # remove samples that have more changes than max_num_changes ot less cnhanges than min_change
        d = npu.distance_arr(instances, self._template)
        random_optimal_instances = instances[(d <= num_changes) & (d >= min_change)]
        random_optimal = npu.not_repeated(known_alternatives, random_optimal_instances)
        return random_optimal

    def generate_initial_neighbours(self):
        # generate normal distribution
        sample_size = 100
        rng = default_rng()
        # calculate mean and std based on min and max values
        means = np.mean([self._max_values, self._min_values], axis=0)
        stds = np.sqrt(self._max_values - means)
        # for each feature of initial_instance, generate samples in normal distribution
        features = [np.round(rng.normal(self._template[k], stds[k], size=sample_size)) for k in range(len(means))]
        # clip for min and max values
        for key, feature_samples in enumerate(features):
            features[key] = np.clip(feature_samples, self._min_values[key], self._max_values[key])
        random_instances = np.array(features).transpose()
        return random_instances

    def generate_neighbours_arr(self, origin_instances, known_alternatives):
        total_neighbours = np.array([], dtype=np.int64).reshape(0, self._template.shape[0])
        for origin_instance in origin_instances:
            neighbours = self.generate_neighbours(origin_instance, known_alternatives)
            total_neighbours = npu.unique_concatenate(total_neighbours, neighbours)
        return total_neighbours

    def generate_neighbours(self, origin_instance, known_alternatives):
        # print("Generating neighbours for: {}, initial instance: {}".format(origin_instance, self._template))
        # generate indexes for increase/decrease movement
        increase_index = self._template > origin_instance
        decrease_index = self._template < origin_instance
        # create movement boundaries array, with maximum distance consideration
        movement_array = origin_instance.copy()
        movement_array[decrease_index] = np.maximum(origin_instance[decrease_index] - NEIGHBOUR_MAX_DEGREE,
                                                    self._template[decrease_index])
        movement_array[increase_index] = np.minimum(origin_instance[increase_index] + NEIGHBOUR_MAX_DEGREE,
                                                    self._template[increase_index])

        # create ranges for each feature exploration
        def ranges(a, b):
            top = max(a, b)
            bottom = min(a, b)
            return np.arange(bottom, top + 1)

        features_movement_range = list(map(ranges, origin_instance, movement_array))
        # create all combinations for each feature movement possible values
        neighbours = cartesian(features_movement_range)
        distances = npu.distance_arr(neighbours, origin_instance)
        neighbours = neighbours[distances <= NEIGHBOUR_MAX_DEGREE]
        unique_neighbours = npu.not_repeated(known_alternatives, neighbours)
        return unique_neighbours
