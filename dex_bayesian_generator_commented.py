from numpy.random import random_sample
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

def run_generator(model, random_alternatives, dataconstraints, initial_instance, target):    
    #TODO: include logging library for log
    print('-----Starting------')
    print('model:',model,'target:',target,'template:',initial_instance)
    
    #TODO: make it read from a config file
    ## --- CONFIG ---
    random_sample_size= 10000            #random instances for the bayesian model
    num_epochs = 150                    #overall number of epochs to run the algorithm
    objective_zero_threshold = 3        #generate new samples when the average objective values has not incrased for x epochs
    improvement_zero_threshold = 50     #improvement on amout of epochs to stop without having improvements.
    current_num_changes= 0
    epoch_without_improvement_threshold = 50
    min_epoch_threshold = 100
    oversampling_weight = 10            #TODO should we keep oversampling?
    neighbours_max_degree = 3
    neighborhood_jitter= .75            # giving search space for the solutions we are finding
    changes_jitter = 8                  #TODO: put 20% of feature size
    # --- END CONFIG ---

    # -- COUNTERS ---
    epoch = 0
    objective_zero = 0 #counter - number of epochs without objective improvement
    improvement_zero = 0 #counter - number of epochs improvement being zero
    epoch_without_improvement = 0
    epoch_max_found = 0
    # --- END COUNTERS ---
    
    surrogate_model = RandomForestRegressor(1000,n_jobs=4)
    checker = InstancesChecker(model, surrogate_model, initial_instance, dataconstraints)
    generator = InstancesGenerator(initial_instance, dataconstraints, neighbours_max_degree)

    # --- BOOTSTRAP ---
    #here we should generate the neighbours of the initial instance
    X = random_alternatives
    Y = checker.calculate_objective_all(X, target)
    #improvement: oversample X based on score Y
    checker.train_surrogate(X,Y)
    # --- END BOOTSTRAP ---

    best_instance_neighbours = []
    best_instances_pool = []
    promising_alternatives_pool = []

    all_counterfactuals = X[Y>0]
    known_alternatives = X.copy() #known_alternatives to avoid duplicates
    best_instance = X[np.argmax(Y)]
    best_instance_output = max(Y)

    #update best
    if best_instance_output:
        X = np.vstack((X, np.repeat([best_instance], oversampling_weight, axis=0)))
        Y = np.concatenate((Y, np.repeat(best_instance_output, oversampling_weight)))
        
        best_instances_pool.append(list(best_instance))
        best_instance_neighbours = generator.generate_neighbours(best_instance, known_alternatives)        

    current_num_changes = npu.distance(np.array(best_instance),np.array(initial_instance))
    
    # stoh_start_time = time.time()
    # stoh_duration =  time.time() - stoh_start_time  #time to first solution
    # random_alternatives = np.array([[1],[1]])
    
    while epoch < num_epochs:
        print("--- epoch {} ----".format(epoch))
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
                if objective_value>=(best_instance_output*neighborhood_jitter) and objective_value>0.01:
                    # best_x_close_tmp = generator.generate_neighbours(instance,positive_target,1,[],neighbours_max_degree)
                    best_x_close_tmp = generator.generate_neighbours(instance, known_alternatives)
                    best_instance_neighbours.extend(best_x_close_tmp)

                if objective_value>=best_instance_output and objective_value>0.01: #new estimated optimum found
                    best_instance = instance
                    best_instance_output = objective_value

                    if objective_value>best_instance_output: #restart best alternatives
                        best_instances_pool=[]                            
                    if list(best_instance) not in best_instances_pool:
                        best_instances_pool.append(list(best_instance))

                    epoch_max_found = epoch
                    epoch_without_improvement = 0
                    improvement_zero=1 #restart counter
                    
                    instance = np.repeat([instance],oversampling_weight,axis=0) #oversample
                    objective_value = np.repeat(objective_value,oversampling_weight) #oversample
                    
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
                len(best_instances_pool),
                len(promising_alternatives_pool)
            ))

        # LAST ITERATION
        # update the model
        if (epoch_without_improvement>=epoch_without_improvement_threshold or epoch==num_epochs) and epoch>min_epoch_threshold:
            print("Last iteration, checking promising alternatives with wiser surrogate model...")
            print("Best current alternative:",best_instance)
            
            threshold = .9
            promising_alternatives_pool.extend(best_instances_pool)
            # final check on the last iteration, with the wiser surrogate model.
            final_alternatives, _ = checker.check_promising_alternatives(
                promising_alternatives_pool,
                best_instance_output,
                best_instances_pool,
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
