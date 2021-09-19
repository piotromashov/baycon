import numpy_utils as npu
import numpy as np

class InstancesGenerator:
    def __init__(self, template, min_values, max_values):
        self._template = template
        self._min_values = min_values
        self._max_values = max_values

    def getRandom(self, size, num_changes, improvement):
        print("Generating alternatives from template.")
        features_amount = len(self._template)
        if num_changes>features_amount/2:
            num_changes = num_changes//2
            
        min_change=1
        if not num_changes:
            num_changes=min_change
        print("num_changes:",num_changes,'min_change:',min_change)

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
        d = npu.distance_arr(instances,self._template,improvement)
        return instances[(d<=num_changes) & (d>=min_change)]

    def getNeighbours():
        pass

    def update(self, known_instances, new_instances):
        last_idx = len(known_instances)
        instances = np.vstack((known_instances, new_instances))
        _,idx_arr = np.unique(instances,axis=0,return_index=True)
        idx_arr = idx_arr[idx_arr>last_idx]
        instances = instances[idx_arr]

        print("Done. Instances size:",instances.shape)
        return instances