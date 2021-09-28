from dex_bayesian_generator import surrogate
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import acquisition_functions as acq_functions
import warnings
import numpy_utils as npu
warnings.filterwarnings("ignore")
import time
from InstancesGenerator import *
from InstancesChecker import *

#returns mean values and standard deviation calculated over the predictions
#from each separate model from a given ensemble models
def get_ensemble_scores(ens_model,X):
    ens_predictions = []
    for est in range(len(ens_model.estimators_)): 
        ens_predictions.append(ens_model.estimators_[est].predict(X))
    ens_predictions = np.array(ens_predictions)

    mu = ens_predictions.mean(axis=0)
    std = ens_predictions.std(axis=0)
    return mu, std

# returns scores caclulated with an acquisition function (see acqusition_functions.py)
def acquisition(model, X, X_candidates):
    mu, _ = get_ensemble_scores(model,X)
    best_mu = max(mu)
    mu, std = get_ensemble_scores(model,X_candidates)
    score = acq_functions.EI(mu, std, best_mu, epsilon=.001)
    return score

# select top_n alternatives based on the acquisition function
def opt_acquisition(model, X, X_candidates, top_n = 10):
    # calculate the acquisition function for each candidate
    scores = acquisition(model, X, X_candidates)
    # locate the index of the largest scores
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

def run_generator(model, random_alternatives, dataconstraints, initial_instance, target, first_sample = 3):    
    #TODO: include logging library for log
    print('-----Starting------')
    print('model:',model,'first_sample:',first_sample,'target:',target,'template:',initial_instance)
    
    #TODO: make it read from a config file
    random_sample_size= 10000            #random instances for the bayesian model
    num_epochs = 150                    #overall number of epochs to run the algorithm
    objective_zero_threshold = 3        #generate new samples when the average objective values has not incrased for x epochs
    improvement_zero_threshold = 50     #improvement on amout of epochs to stop without having improvements.
    current_num_changes= 0
    epoch_without_improvement_threshold = 50
    min_epoch_threshold = 100
    oversampling_weight = 10            #TODO should we keep oversampling?
    neighbours_max_degree = 3
    
    #Generate starting dataset and train the surrogate_model
    generator = InstancesGenerator(initial_instance, dataconstraints, neighbours_max_degree)
    checker = InstancesChecker(model, RandomForestRegressor(1000,n_jobs=4), initial_instance, dataconstraints)

    # objective function, for now we work with this single number (single objective optimization)
    X = random_alternatives
    Y=checker.calculate_objective_all(X, target)
    
    checker.train_surrogate(X,Y)
    known_alternatives = X.copy() #known_alternatives to avoid duplicates
    
    best_instance = X[np.argmax(Y)]
    best_instance_neighbours = []

    best_Y = max(Y)
    if best_Y>0:
        current_best = X[np.argmax(Y)]
        #oversample current best if it carries information
        current_best = np.repeat([current_best],oversampling_weight,axis=0)
        X = np.vstack((X, current_best))
        tmp_Y = best_Y
        tmp_Y = np.repeat(tmp_Y,oversampling_weight)
        Y = np.concatenate((Y, tmp_Y))
        
        #save best_x neighbours to be checked in the next iteration
        best_instance = current_best[0]
        best_instance_neighbours = generator.generate_neighbours(best_instance, known_alternatives)        
        # neighbour_instances = generator.generate_neighbours(best_instance,positive_target,1,[],neighbours_max_degree)

    current_num_changes = npu.distance(np.array(best_instance),np.array(initial_instance))
    
    epoch = 0
    objective_zero = 0 #counter - number of epochs without objective improvement
    improvement_zero = 0 #counter - number of epochs improvement being zero
    epoch_without_improvement = 0
    epoch_max_found = 0
    
    # stoh_start_time = time.time()
    # stoh_duration =  time.time() - stoh_start_time  #time to first solution
    random_alternatives = np.array([[1],[1]])
    
    # giving search space for the solutions we are finding
    neighborhood_jitter= .75
    
    changes_jitter = 8   #TODO: put 20% of feature size
    print('neighborhood_jitter',neighborhood_jitter)
    print('neighbours_max_degree',neighbours_max_degree)
    
    best_alternatives_pool = []
    best_alternatives_pool.append(list(best_instance))
    
    promising_alternatives_pool = []
    promising_alternatives_pool.extend(best_instance_neighbours)

    all_counterfactuals = X[Y==target]
    
    while epoch < num_epochs:
        print('----')
        print(epoch,'epoch')
        #helper variables
        objective_zero += 1
        improvement_zero += 1
        epoch += 1
        epoch_without_improvement += 1
        
        #check if we have close neighbours to be checked
        print('neighbours to be checked:',len(best_instance_neighbours))
        if len(best_instance_neighbours):
            #obtain top 10 of the neighbours (based on the acquisition function)
            top_instances,_ = opt_acquisition(
                checker.surrogate(), #the bayesian model that is used
                X, #possible counterfactuals with known objective value
                np.array(best_instance_neighbours), #neighbouring instances close to the current best
            )
            print('predicting neighbours...')
            estimated_values = checker.surrogate().predict(top_instances)
            best_instance_neighbours = []
        else:
            #go with random counterfactuals
            if len(random_alternatives) < 10:
                # how many features we allow to change
                num_changes=changes_jitter+(current_num_changes-1)
                random_alternatives = generator.generate_random(random_sample_size, num_changes, known_alternatives)
                
            if len(random_alternatives):
                top_instances,random_alternatives = opt_acquisition(checker.surrogate(), X,random_alternatives)
                print('predicting random alternatives...')
                estimated_values = checker.surrogate().predict(top_instances)
    
        if len(top_instances):
            print('calculating objective...')
            objective_values = checker.calculate_objective_all(top_instances, target)
            counterfactuals = top_instances[objective_values>0.01]
            all_counterfactuals = npu.stack_not_repeated(all_counterfactuals, counterfactuals)

            #add to known_alternatives
            known_alternatives = npu.stack_not_repeated(known_alternatives,top_instances)

            print("predicted | actual")
            for k, v in enumerate(estimated_values):
                print('instance {}:  {}, {}'.format(k+1, np.round(estimated_values[k],3), np.round(objective_values[k],3)) )
            print('current optimal: ', np.round(max(Y),3))

            # add the data to the dataset for the surogate model
            for k in range(len(objective_values)):
                objective_value = objective_values[k]
                instance = top_instances[k]

                if objective_value>0.01:
                    objective_zero=0 #restart counter

                # check if close to current best, then it's candidate to keep exploring
                if objective_value>=(best_Y*neighborhood_jitter) and objective_value>0.01:
                    # best_x_close_tmp = generator.generate_neighbours(instance,positive_target,1,[],neighbours_max_degree)
                    best_x_close_tmp = generator.generate_neighbours(instance, known_alternatives)
                    best_instance_neighbours.extend(best_x_close_tmp)

                if objective_value>=best_Y and objective_value>0.01: #new estimated optimum found
                    best_instance = instance
                    if objective_value>best_Y: #restart best alternatives
                        best_alternatives_pool=[]                            
                    if list(best_instance) not in best_alternatives_pool:
                        best_alternatives_pool.append(list(best_instance))
                    epoch_without_improvement = 0
                    epoch_max_found = epoch


                    improvement_zero=1 #restart counter
                    
                    instance = np.repeat([instance],oversampling_weight,axis=0) #oversample
                    objective_value = np.repeat(objective_value,oversampling_weight) #oversample
                    best_Y = objective_value[0]
                    
                Y = np.concatenate((Y, objective_value), axis=None)
                X = np.vstack((X, instance))

            top_instances = []

            # retrain our bayesian model for improvement on new information.
            print("re-training surrogate model with data size: {}".format(len(X)))
            checker.train_surrogate(X,Y)
            
            if len(best_instance_neighbours)>0:
                #start trusting the model on the promising alternatives after some experience.
                if epoch>3: #store promising_alternatives_pool after nth learning epoch
                    promising_alternatives_pool.extend(best_instance_neighbours)
            # alternatives to be check after the last iteration, because the Bayesian model may be wiser then.
            print("known alternatives size {}\nbest found in epoch: {}, best neighbouring instances: {}, best alternatives pool: {}, promising alternatives: {}".format(
                known_alternatives.shape[0],
                epoch_max_found,
                len(best_instance_neighbours),
                len(best_alternatives_pool),
                len(promising_alternatives_pool)
            ))

        # LAST ITERATION
        # update the model
        if (epoch_without_improvement>=epoch_without_improvement_threshold or epoch==num_epochs) and epoch>min_epoch_threshold:
            print("Last iteration, checking promising alternatives with wiser surrogate model...")
            print("Best current alternative:",best_instance)
            
            threshold = .9
            promising_alternatives_pool.extend(best_alternatives_pool)
            # final check on the last iteration, with the wiser surrogate model.
            final_alternatives, _ = checker.check_promising_alternatives(
                promising_alternatives_pool,
                best_Y,
                best_alternatives_pool,
                threshold,
                target
            )
            break

        objective_zero_bool = objective_zero_threshold<=objective_zero
        objective_improvement_bool = improvement_zero_threshold<=improvement_zero
            
        #distance in the feature space from initial instance and the optimum solution
        num_changes = npu.distance(np.array(best_instance),np.array(initial_instance))
        changes_improvement_bool = current_num_changes>num_changes-1
        
        #check if we need to update the pool of random alternatives because we run out of neighbours and
        # the current pool is small OR
        # there have'nt been any improvement in the past K epochs OR
        # the objective function is stuck at 0 (no improvements on optimum)
        if len(best_instance_neighbours)==0 and (changes_improvement_bool or objective_zero_bool or objective_improvement_bool):
            current_num_changes = num_changes
            num_changes=changes_jitter+current_num_changes
 
            random_alternatives = generator.generate_random(random_sample_size, num_changes, known_alternatives)

            objective_zero=0
            improvement_zero = 0    
            
    final_alternatives = npu.stack_not_repeated(all_counterfactuals, final_alternatives)
    return final_alternatives, 0
