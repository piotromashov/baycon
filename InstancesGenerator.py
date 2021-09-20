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

    #recoursive function
    #TODO: improve this function, remove positive distances
    #probably can be optimized
    #generates neighbours (i.e., similar solutions with same or samlled distance values than a given alternative x
    #with respect to a templace
    #max_degree tells hown many times the function will be called
    #max_degree=1 (only x's neighbours)
    #max_degree=2 (x's neighbours and neighbours of x's neighbours)
    #max_degree=3 (x's neighbours, neighbours of x's neighbours, and neighbours of the neighbours of x's neighbours)
    def generate_neighbours(self, x,positive_target,current_degree=1,x_close=[],max_degree=2):
        idx_arr = npu.find_changes(x,self._template,positive_target)
        while current_degree<max_degree and len(idx_arr)!=0 and len(x_close)<5000:
            current_degree = current_degree+1
            for position in idx_arr:
                tmp_x = x.copy()
                if positive_target: #this update will dicrease the distance function
                    tmp_x[position] = tmp_x[position]-1
                else:
                    tmp_x[position] = tmp_x[position]+1
                x_close.append(tmp_x)
                
                if positive_target: #this update will not change the distance function
                    idx_arr_neg  = np.argwhere(tmp_x < self._template)
                    values = tmp_x[idx_arr_neg]+1
                else:
                    idx_arr_neg  = np.argwhere(tmp_x> self._template)
                    values = tmp_x[idx_arr_neg]-1
                if len(idx_arr_neg)>0:
                    tmp_arr = np.repeat([tmp_x],len(idx_arr_neg),axis=0)
                    tmp_arr =npu.update_per_row(tmp_arr,idx_arr_neg.flatten(),values)
                    x_close.extend(tmp_arr)
                
                self.generate_neighbours(tmp_x, positive_target, current_degree, x_close, max_degree) 
        #remove duplicates
        if len(x_close)>0:
            if len(x_close)>1:
                x_close = np.unique(x_close,axis=0)
            return list(x_close)
        return []

def remove_duplicates(known_instances, new_instances):
    last_idx = len(known_instances)
    instances = np.vstack((known_instances, new_instances))
    _,idx_arr = np.unique(instances,axis=0,return_index=True)
    idx_arr = idx_arr[idx_arr>last_idx]
    instances = instances[idx_arr]

    return instances

