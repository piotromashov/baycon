import numpy as np
import numpy_utils as npu

class InstancesChecker:
    def __init__(self, objective_model, surrogate_model, initial_instance, dataconstraints):
        self._objective_model = objective_model
        self._surrogate_model = surrogate_model
        self._initial_instance = initial_instance
        self._dataconstraints = dataconstraints

    def train_surrogate(self, X, Y):
        self._surrogate_model.fit(X,Y)

    def surrogate(self):
        return self._surrogate_model

    #check promising_alternatives_pool 
    def check_promising_alternatives(self, alternatives,best_y,best_pool,threshold,target):
        print('checking promising_alternatives')
        top_n = 1000
        if not len(alternatives):
            alternatives = np.array(best_pool)[:top_n]
        else:
            #predict the optimization function with the surogate model
            if np.array(alternatives).ndim==1:
                pred = self._surrogate_model.predict(np.array(alternatives).reshape(1, -1))
            else:
                pred = self._surrogate_model.predict(alternatives)
            alternatives = np.array(alternatives)

            #get n-top ranked alternatives
    
            alternatives = alternatives[pred>=best_y*threshold]
            pred = pred[pred>=best_y*threshold]
            if len(alternatives)>top_n:
                ix = np.argpartition(pred, -top_n)[-top_n:] #get top_n candidates
                alternatives= alternatives[ix]

        Y = np.repeat(best_y,len(best_pool))
        #for each top-ranked alternative
        #check real objective value of the alternative and its neighbours
        if len(alternatives)>0:
            #check real objective value of all alternatives
            Y_tmp=self.calculate_objective_all(alternatives, target)
            
            alternatives = np.vstack((alternatives,best_pool))
            Y = np.concatenate((Y_tmp,Y))
        else:
            alternatives = np.array(best_pool)

        alternatives =  alternatives[[Y>=best_y]]
        Y = Y[[Y>=best_y]]

        #generate neighbours
        x_close = []
        alternatives_tmp = alternatives
        for i in range(10):
            len_before = len(x_close) 
            for tmp_x in alternatives_tmp:
                #TODO: change whole function to generate neighbours, here we had positive_target, which is no longer used
                if True: #this updaee would not change the distance value between the template and the alternative
                    idx_arr_neg  = np.argwhere(tmp_x<self._initial_instance)
                    values = tmp_x[idx_arr_neg]+1
                else:
                    idx_arr_neg  = np.argwhere(tmp_x>self._initial_instance)
                    values = tmp_x[idx_arr_neg]-1
                if len(idx_arr_neg)>0:
                    tmp_arr = np.repeat([tmp_x],len(idx_arr_neg),axis=0)
                    tmp_arr =npu.update_per_row(tmp_arr,idx_arr_neg.flatten(),values)
                    x_close.extend(tmp_arr)
            # get the last iteration discovered neighbours
            alternatives_tmp = x_close[len_before:]
            if len(x_close)==len_before or len(x_close)>top_n: #ro avoid memmory error
                break
        #check the real objective value of the neighbours
        if len(x_close)>0:
            x_close = list(npu.not_repeated(alternatives, x_close))
            
            if len(x_close)>0:  
                Y_tmp=self.calculate_objective_all(x_close, target)
                alternatives = np.vstack((alternatives,x_close))
                Y = np.concatenate((Y,Y_tmp))
                alternatives = alternatives[Y>=best_y]
                Y = Y[Y>=best_y]
        alternatives = np.unique(alternatives,axis=0)
        return alternatives,Y

    def promising_pool_check(self, alternatives, best_instances, best_output, target):
        objective_values = self.calculate_objective_all(alternatives, target)
        #update new best_output if objective_values has a greater one
        #train surrogate model with new known outputs
        #run acquisition function and get output sorted

        #run generation of neighbours for each one of them

        pass

        #calculate objective fuction for a list aletrnatives
    def calculate_objective_all(self, alternatives, target):
        #get model prediction on those values
        Y = self._objective_model.predict(alternatives)
        Y = np.array(Y)

        max_distance = self._dataconstraints.features_max_distance()

        # here should go the cost of attribute changes and their weights
        instance_distance = npu.distance_arr(alternatives, self._initial_instance)

        # closeness to feature space of the potential counterfactual to the initial instance.
        relative_similarity = 1-instance_distance/max_distance

        # check if we are moving towards the target or not.
        #if we are not moving towards the target, this is weighted as 0
        targets_achieved = Y==target
        objective_values = relative_similarity*targets_achieved    
        return objective_values