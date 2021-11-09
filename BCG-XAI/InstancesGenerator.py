import numpy as np

import numpy_utils as npu


class InstancesGenerator:
    def __init__(self, template, data_analyzer):
        self._initial_instance = template
        self._min_values = data_analyzer.min_feature_values()
        self._max_values = data_analyzer.max_feature_values()
        self._distance_calculator = data_analyzer.distance_calculator()

    def generate_random(self, distance, known_alternatives):
        instances = npu.uniform_dist_sample(self._min_values, self._max_values)
        # remove samples that are same as the template
        instances = instances[np.sum(instances != self._initial_instance, axis=1) > 0]
        distances = self._distance_calculator.gower(self._initial_instance, instances)
        instances = instances[distances <= distance]
        instances = npu.not_repeated(known_alternatives, instances)
        return instances

    def generate_initial_neighbours(self):
        # calculate mean and std based on min and max values
        means = np.mean([self._max_values, self._min_values], axis=0)
        sds = np.sqrt(self._max_values - means)
        neighbours = npu.normal_dist_sample(self._initial_instance, sds, self._min_values, self._max_values)
        return neighbours

    def generate_neighbours_arr(self, origin_instances, known_alternatives):
        total_neighbours = np.array([], dtype=np.int64).reshape(0, self._initial_instance.shape[0])
        for origin_instance in origin_instances:
            neighbours = self.generate_neighbours(origin_instance, known_alternatives)
            total_neighbours = npu.unique_concatenate(total_neighbours, neighbours)
        return total_neighbours

    def generate_neighbours(self, origin_instance, known_alternatives):
        # calculate mean and std based on min and max values
        means = origin_instance
        sds = np.sqrt(np.abs(self._initial_instance - means))
        # generate indexes for top/bottom values
        increase_index = self._initial_instance >= origin_instance
        decrease_index = self._initial_instance < origin_instance
        tops = np.where(increase_index, self._initial_instance, origin_instance)
        bottoms = np.where(decrease_index, self._initial_instance, origin_instance)

        neighbours = npu.normal_dist_sample(means, sds, bottoms, tops)

        # remove all neighbours with distances that are bigger than current origin instance to the initial one
        distances = self._distance_calculator.gower(origin_instance, neighbours)
        distance_from_origin_to_initial = self._distance_calculator.gower(self._initial_instance,
                                                                          np.array([origin_instance]))
        neighbours = neighbours[distances <= distance_from_origin_to_initial]
        unique_neighbours = npu.not_repeated(known_alternatives, neighbours)
        return unique_neighbours
