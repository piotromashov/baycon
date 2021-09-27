import numpy_utils as npu
import numpy as np
from sklearn.utils.extmath import cartesian

class InstancesGenerator:
    def __init__(self, template, dataconstraints):
        self._template = template
        self._min_values = dataconstraints.min_feature_values()
        self._max_values = dataconstraints.max_feature_values()

    def getRandom(self, size, num_changes, improvement):
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
        print("Generated random instances {}".format(random_optimal_instances.shape[0]))
        return random_optimal_instances


    # def neighbours_breadth_generation(self, origin_instance, breadth_exploration_degree = 3):
    #     total_neighbourhood = origin_instance
    #     #get first level of neighbourhood zone
    #     neighbourhood_zone = [origin_instance]
    #     for _ in range(breadth_exploration_degree):
    #         #neighbours to explore, get previous explored zone
    #         neighbours = neighbourhood_zone
    #         #clean slate for new level of neighbourhood
    #         neighbourhood_zone = [] 
    #         for neighbour in neighbours:
    #             # get closer instances to destination (aka neighbourhood zone)
    #             neighbourhood_zone.extend(self.generate_closer_instances(neighbour))
    #         #add new neighbourhood zone into total_neighbours
    #         if neighbourhood_zone:
    #             total_neighbourhood = np.vstack((total_neighbourhood, neighbourhood_zone))
    #         #remove duplicates
    #         total_neighbourhood = np.unique(total_neighbourhood,axis=0)

    #         if len(total_neighbourhood)>5000:
    #             break
    #     # print("Generated neighbours: {}".format(len(total_neighbourhood)))
    #     return total_neighbourhood
            

    # def generate_closer_instances(self, origin_instance):
    #     closer_instances = []
    #     for k, _ in enumerate(self._template):
    #         if origin_instance[k]==self._template[k]:
    #             continue
    #         auxiliar_instance = origin_instance.copy()
    #         auxiliar_instance[k] += 1 if origin_instance[k]<self._template[k] else -1
    #         closer_instances.append(auxiliar_instance)
    #     return closer_instances

    def generate_neighbours(self, origin_instance, max_distance = 3):
        print("Generating neighbours for: {}, initial instance: {}".format(origin_instance, self._template))
        #generate indexes for increase/decrease movement
        increase_index = self._template>origin_instance
        decrease_index = self._template<origin_instance
        #create movement boundaries array, with maximum distance consideration
        movement_array = origin_instance.copy()
        movement_array[decrease_index] = np.maximum(origin_instance[decrease_index]-max_distance, self._template[decrease_index])
        movement_array[increase_index] = np.minimum(origin_instance[increase_index]+max_distance, self._template[increase_index])

        #create ranges for each feature exploration
        def ranges(a, b):
            top = max(a, b)
            bottom = min(a, b)
            return np.arange(bottom, top+1)

        features_movement_range = list(map(ranges, origin_instance, movement_array))
        #create all combinations for each feature movement possible values
        instances = cartesian(features_movement_range)
        distances = npu.distance_arr(instances, origin_instance)
        instances = instances[distances<=max_distance]
        print("Generated neighbours instances: {}".format(instances.shape[0]))
        return instances

    #recoursive function
    #TODO: improve this function, remove positive distances
    #probably can be optimized
    #generates neighbours (i.e., similar solutions with same or samlled distance values than a given alternative x
    #with respect to a templace
    #max_degree tells hown many times the function will be called
    #max_degree=1 (only x's neighbours)
    #max_degree=2 (x's neighbours and neighbours of x's neighbours)
    #max_degree=3 (x's neighbours, neighbours of x's neighbours, and neighbours of the neighbours of x's neighbours)
    # def generate_neighbours(self, x,positive_target,current_degree=1,x_close=[],max_degree=2):
    #     idx_arr = npu.find_changes(x,self._template)
    #     while current_degree<max_degree and len(idx_arr)!=0 and len(x_close)<5000:
    #         current_degree = current_degree+1
    #         for position in idx_arr:
    #             tmp_x = x.copy()
    #             if positive_target: #this update will dicrease the distance function
    #                 tmp_x[position] = tmp_x[position]-1
    #             else:
    #                 tmp_x[position] = tmp_x[position]+1
    #             x_close.append(tmp_x)
                
    #             if positive_target: #this update will not change the distance function
    #                 idx_arr_neg  = np.argwhere(tmp_x < self._template)
    #                 values = tmp_x[idx_arr_neg]+1
    #             else:
    #                 idx_arr_neg  = np.argwhere(tmp_x> self._template)
    #                 values = tmp_x[idx_arr_neg]-1
    #             if len(idx_arr_neg)>0:
    #                 tmp_arr = np.repeat([tmp_x],len(idx_arr_neg),axis=0)
    #                 tmp_arr =npu.update_per_row(tmp_arr,idx_arr_neg.flatten(),values)
    #                 x_close.extend(tmp_arr)
                
    #             self.generate_neighbours(tmp_x, positive_target, current_degree, x_close, max_degree) 
    #     #remove duplicates
    #     if len(x_close)>0:
    #         if len(x_close)>1:
    #             x_close = np.unique(x_close,axis=0)
    #         return list(x_close)
    #     return []

