import numpy_utils as npu
import numpy as np
from sklearn.utils.extmath import cartesian

class InstancesGenerator:
    def __init__(self, template, dataconstraints, neighbours_max_degree):
        self._template = template
        self._min_values = dataconstraints.min_feature_values()
        self._max_values = dataconstraints.max_feature_values()
        self._max_distance = neighbours_max_degree

    def generate_random(self, size, num_changes, known_alternatives):
        features_amount = len(self._template)
        if num_changes>features_amount/2:
            num_changes = num_changes//2
            
        min_change=1
        if not num_changes:
            num_changes=min_change

        instances = np.random.rand(size,len(self._template)) #random matrix with n samples with values between 0 and 1
        
        #maximum and minimum allowed change for one attribute (e.g., from 0 to 3)
        max_value = max(self._max_values) 
        min_value = -max_value

        random_matrix = np.random.randint(min_value,max_value,(size, features_amount))

        instances = random_matrix+self._template #increase each row with the template (i.e., increase template values by zero,one or two)
        
        #remove values bigger than max value
        while len(instances[instances > self._max_values]) > 0:
            instances[instances > self._max_values] = instances[instances > self._max_values] - 1
        instances[instances < self._min_values] = 0 

        instances = instances.astype(int)
        
        #increase atrute values that have lower values thetemplate
        
        #remove samples that are same as the template
        instances=instances[np.sum(instances != self._template, axis = 1) > 0] 

        #remove samples that have more changes than max_num_changes ot less cnhanges than min_change
        d = npu.distance_arr(instances, self._template)
        random_optimal_instances = instances[(d<=num_changes) & (d>=min_change)]
        random_optimal = npu.not_repeated(known_alternatives, random_optimal_instances)
        return random_optimal

    def generate_neighbours_arr(self, origin_instances, known_alternatives):
        total_neighbours = np.array([], dtype=np.int64).reshape(0, self._template.shape[0])
        for origin_instance in origin_instances:
            neighbours = self.generate_neighbours(origin_instance, known_alternatives)
            total_neighbours = npu.unique_concatenate(total_neighbours,neighbours)
        return total_neighbours

    def generate_neighbours(self, origin_instance, known_alternatives):
        # print("Generating neighbours for: {}, initial instance: {}".format(origin_instance, self._template))
        #generate indexes for increase/decrease movement
        increase_index = self._template>origin_instance
        decrease_index = self._template<origin_instance
        #create movement boundaries array, with maximum distance consideration
        movement_array = origin_instance.copy()
        movement_array[decrease_index] = np.maximum(origin_instance[decrease_index]-self._max_distance, self._template[decrease_index])
        movement_array[increase_index] = np.minimum(origin_instance[increase_index]+self._max_distance, self._template[increase_index])

        #create ranges for each feature exploration
        def ranges(a, b):
            top = max(a, b)
            bottom = min(a, b)
            return np.arange(bottom, top+1)

        features_movement_range = list(map(ranges, origin_instance, movement_array))
        #create all combinations for each feature movement possible values
        neighbours = cartesian(features_movement_range)
        distances = npu.distance_arr(neighbours, origin_instance)
        neighbours = neighbours[distances<=self._max_distance]
        unique_neighbours = npu.not_repeated(known_alternatives, neighbours)
        return unique_neighbours