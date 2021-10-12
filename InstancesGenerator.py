from numpy.random import default_rng
import numpy_utils as npu
import numpy as np
from scipy.stats import truncnorm
from Distance import Distance

RANDOM_SAMPLE_SIZE = 10000
NEIGHBOURS_SAMPLE_SIZE = 100


class RandomDistMocker:
    def __init__(self, value):
        self._value = value

    def rvs(self, sample_size):
        return np.repeat(self._value, sample_size)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    if sd < 1 or low == upp:
        return RandomDistMocker(mean)
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def normal_dist_neighbours(means, sds, bottoms, tops):
    normal_distributions = [get_truncated_normal(means[k], sds[k], bottoms[k], tops[k]) for k in range(len(means))]
    features = [np.round(nd.rvs(NEIGHBOURS_SAMPLE_SIZE)) for nd in normal_distributions]
    neighbours = np.array(features).transpose()
    return neighbours


class InstancesGenerator:
    def __init__(self, template, data_constraints):
        self._template = template
        self._min_values = data_constraints.min_feature_values()
        self._max_values = data_constraints.max_feature_values()
        self._distance_calculator = Distance(data_constraints)

    def generate_random(self, distance, known_alternatives):
        features_amount = len(self._template)
        # maximum and minimum allowed change for one attribute (e.g., from 0 to 3)
        max_value = max(self._max_values)
        min_value = -max_value

        random_matrix = np.random.randint(min_value, max_value, (RANDOM_SAMPLE_SIZE, features_amount))
        # increase each row with the template (i.e., increase template values by zero,one or two)
        instances = random_matrix + self._template
        # remove values bigger than max value
        while len(instances[instances > self._max_values]) > 0:
            instances[instances > self._max_values] = instances[instances > self._max_values] - 1
        instances[instances < self._min_values] = 0

        instances = instances.astype(int)
        # remove samples that are same as the template
        instances = instances[np.sum(instances != self._template, axis=1) > 0]

        distances = self._distance_calculator.gower(self._template, instances)
        random_optimal_instances = instances[distances <= distance]
        random_optimal = npu.not_repeated(known_alternatives, random_optimal_instances)
        return random_optimal

    def generate_initial_neighbours(self):
        # calculate mean and std based on min and max values
        means = np.mean([self._max_values, self._min_values], axis=0)
        sds = np.sqrt(self._max_values - means)
        neighbours = normal_dist_neighbours(self._template, sds, self._min_values, self._max_values)
        return neighbours

    def generate_neighbours_arr(self, origin_instances, known_alternatives):
        total_neighbours = np.array([], dtype=np.int64).reshape(0, self._template.shape[0])
        for origin_instance in origin_instances:
            neighbours = self.generate_neighbours(origin_instance, known_alternatives)
            total_neighbours = npu.unique_concatenate(total_neighbours, neighbours)
        return total_neighbours

    def generate_neighbours(self, origin_instance, known_alternatives):
        # calculate mean and std based on min and max values
        means = origin_instance
        sds = np.sqrt(np.abs(self._template - means))
        # generate indexes for top/bottom values
        increase_index = self._template >= origin_instance
        decrease_index = self._template < origin_instance
        tops = np.where(increase_index, self._template, origin_instance)
        bottoms = np.where(decrease_index, self._template, origin_instance)

        neighbours = normal_dist_neighbours(means, sds, bottoms, tops)

        # remove all neighbours with distances that are bigger than current origin instance to the initial one
        distances = self._distance_calculator.gower(origin_instance, neighbours)
        distance_from_origin_to_initial = self._distance_calculator.gower(self._template, np.array([origin_instance]))
        neighbours = neighbours[distances <= distance_from_origin_to_initial]
        unique_neighbours = npu.not_repeated(known_alternatives, neighbours)
        return unique_neighbours
