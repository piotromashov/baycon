import numpy as np
import numpy_utils as npu
import acquisition_functions as acq_functions

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

    def rank(self, known_instances, instances_to_check):
        top_instances, _ = self.opt_acquisition(known_instances, instances_to_check)
        return top_instances
        

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

    #returns mean values and standard deviation calculated over the predictions
    #from each separate model from a given ensemble models
    def get_ensemble_scores(self, X):
        ens_predictions = []
        for est in range(len(self._surrogate_model.estimators_)): 
            ens_predictions.append(self._surrogate_model.estimators_[est].predict(X))
        ens_predictions = np.array(ens_predictions)

        mu = ens_predictions.mean(axis=0)
        std = ens_predictions.std(axis=0)
        return mu, std

    # returns scores caclulated with an acquisition function (see acqusition_functions.py)
    def acquisition(self, X, X_candidates):
        mu, _ = self.get_ensemble_scores(X)
        best_mu = max(mu)
        mu, std = self.get_ensemble_scores(X_candidates)
        score = acq_functions.EI(mu, std, best_mu, epsilon=.001)
        return score

    # select top_n alternatives based on the acquisition function
    def opt_acquisition(self, X, X_candidates, top_n = 10):
        # calculate the acquisition function for each candidate
        scores = self.acquisition(X, X_candidates)
        # locate the index of the largest scores
        #TODO: remove this safeguard
        if top_n>len(scores)//2:
            top_n = len(scores)//2
        best_alternatives_index = np.argpartition(scores, -top_n)[-top_n:] #get top_n candidates
        best_alternatives= X_candidates[best_alternatives_index]
        random_alternatives_index =np.random.randint(0,len(X_candidates),top_n) #get_random_candidates
        random_alternatives =  X_candidates[random_alternatives_index]
        
        #remove candidates from the random candidates, as they will be available in X
        remove_index = np.concatenate([random_alternatives_index,best_alternatives_index])
        X_candidates = np.delete(X_candidates,remove_index,axis=0)

        alternatives = npu.stack_not_repeated(best_alternatives, random_alternatives)
            
        return alternatives,X_candidates