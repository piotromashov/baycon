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

    #join top-n candidates and n random cadindates and calculate their string representations
    alternatives = np.vstack([best_alternatives,random_alternatives])
        
    return alternatives,X_candidates

#returns min and max values for each attribute
def generate_minMax_constrais(attribute_values):
    #TODO: make this dynamically in base of the feature ranges
    return np.repeat(0,8), np.repeat(10,8)

def run_generator(model, dataset, feature_values, initial_instance, target, neighbours_max_degree=3, first_sample = 3,positive_target = True):
    print("model:",model,'target:',target,'positive_target:',positive_target)
    
    #random instances for the bayesian model
    random_sample_size=10000
    #overall number of epochs to run the algorithm
    num_epochs = 150 
    # num_epochs = 80 
    #generate new samples when the size is lower than x% of the initial size
    random_sample_size_threshold = .1 
    #generate new samples when the average objective values has not incrased for x epochs
    objective_zero_threshold = 3
    #improvement on amout of epochs to stop without having improvements.
    improvement_zero_threshold = 50   
    current_num_changes=0
    new_max_epoch_threshold = 50
    min_epoch_threshold = 100
                           
    print()
    print('-----------')
    surrogate_model = RandomForestRegressor(1000,n_jobs=4)
    
    #Generate starting dataset and train the surrogate_model
    # string representation of the initial instances.
    X = npu.generate_random_alternatives(dataset, n = 10) 

    min_values, max_values = generate_minMax_constrais(feature_values)
    generator = InstancesGenerator(initial_instance, min_values, max_values)
    checker = InstancesChecker(model, surrogate_model, initial_instance)

    # objective function, for now we work with this single number (single objective optimization)
    Y=checker.calculate_objective_all(X, target, positive_target)
    
    checker.train_surrogate(X,Y)
    known_alternatives = X.copy() #known_alternatives to avoid duplicates

    #for log
    print('model:',model,'first_sample:',first_sample,'target:',target,'template:',initial_instance,'positive_target:',positive_target)
    
    #oversample current best if it carries some information
    best_x = X[np.argmax(Y)]
    best_x_close = []

    best_Y = max(Y)
    if best_Y>0:
        current_best = X[np.argmax(Y)]
        current_best = np.repeat([current_best],10,axis=0)
        X = np.vstack((X, current_best))
        actual = best_Y
        actual = np.repeat(actual,10)
        Y = np.concatenate((Y, actual))
        
        #save best_x neighbours to be checked in the next iteration
        best_x = current_best[0]                
        best_x_close = generator.generate_neighbours(best_x,positive_target,1,[],neighbours_max_degree)

    current_num_changes = npu.distance(np.array(best_x),np.array(initial_instance),positive_target)-1
    
    i=0
    objective_zero = 0 #counter - number of epochs without objective improvement
    improvement_zero=0 #counter - number of epochs improvement being zero
    
    # estimation of the objective function
    EST_epoch_mean = []
    # actual value of the objective function
    Y_epoch_mean= []
    BEST_alternatives_pool_arr = []
    stoh_start_time = time.time()
    random_alternatives = np.array([[1],[1]])
    
    # giving search space for the solutions we are finding
    neighborhood_jitter= .75
    changes_jitter=8
    print('neighborhood_jitter',neighborhood_jitter)
    print('neighbours_max_degree',neighbours_max_degree)
    
    best_alternatives_pool = []
    best_alternatives_pool.append(list(best_x))
    
    promising_alternatives_pool = []
    promising_alternatives_pool.extend(best_x_close)
    new_max_epoch = 0    
    
    
    while i < num_epochs:
        print('----')
        print(i,'epoch')
        #helper variables
        objective_zero = objective_zero+1
        improvement_zero = improvement_zero+1
        i=i+1
        new_max_epoch = new_max_epoch+1
        
        #check if we have close neighbours to be checked
        print('neighbours to be checked:',len(best_x_close))
        if len(best_x_close)>0:
            #obtain top 10 of the neighbours (based on the acquisition function)
            x_num_arr,_ = opt_acquisition(
                surrogate_model, #the bayesian model that is used
                X, #possible counterfactuals with known objective value
                np.array(best_x_close), #neighbouring instances close to the current best
            )
            print('predicting...')
            est_arr = surrogate_model.predict(x_num_arr)
            best_x_close = []
        else:
            #go with random counterfactuals
            if len(random_alternatives)<10:
                # how many features we allow to change
                num_changes=changes_jitter+current_num_changes
                random_alternatives = generator.getRandom(random_sample_size, num_changes, positive_target)
                random_alternatives = remove_duplicates(known_alternatives, random_alternatives)
                
            if len(random_alternatives)>0:
                x_num_arr,random_alternatives = opt_acquisition(surrogate_model, X,random_alternatives)
                print('predicting...')
                est_arr = surrogate_model.predict(x_num_arr)
    
        if len(x_num_arr)!=0:
            print('calculating objective...')
            actual_arr = []
            actual_arr = checker.calculate_objective_all(x_num_arr, target, positive_target)

            #for log - save average estimated and real objective values 
            if len(est_arr)>10:
                EST_epoch_mean.extend([np.mean(est_arr[:10]),np.mean(est_arr[10:])]) #first 5 are infromative samples second 5 are random samples
                Y_epoch_mean.extend([np.mean(actual_arr[:10]),np.mean(actual_arr[10:])])
            else:
                EST_epoch_mean.extend([np.mean(est_arr),-1]) #first 5 are infromative samples second 5 are random samples
                Y_epoch_mean.extend([np.mean(actual_arr),-1])

            #add to known_alternatives
            known_alternatives = np.vstack([known_alternatives,x_num_arr])

            print('estimated value for instances:\n',np.round(est_arr,3))
            print('actual value for instances:\n',np.round(actual_arr,3))
            print('optimal: ', np.round(max(Y),3))

            # add the data to the dataset for the surogate model
            for k in range(len(actual_arr)):
                actual = actual_arr[k]
                x_sample = x_num_arr[k]

                if actual>0.01:
                    objective_zero=0 #restart counter
                   # improvement_weight = len(X)/10 #oversample improvement instance using improvement_weight
                improvement_weight = 1

                # check if close to current best, then it's candidate to keep exploring
                if actual>=(best_Y*neighborhood_jitter) and actual>0.01:
                    best_x_close_tmp = generator.generate_neighbours(x_sample,positive_target,1,[],neighbours_max_degree)
                    best_x_close.extend(best_x_close_tmp)

                if actual>=best_Y and actual>0.01: #new estimated optimum found
                    best_x = x_sample
                    if actual>best_Y: #restart best alternatives
                        best_alternatives_pool=[]                            
                        BEST_alternatives_pool_arr = list(np.repeat(0,len(BEST_alternatives_pool_arr)))
                    if list(best_x) not in best_alternatives_pool:
                        best_alternatives_pool.append(list(best_x))
                    new_max_epoch = 0

                    improvement_zero=1 #restart counter
                    
                    # maybe this does not oversample? check.
                    x_sample = np.repeat([x_sample],improvement_weight,axis=0) #oversample
                    actual = np.repeat(actual,improvement_weight) #oversample
                    Y = np.concatenate((Y, actual))
                    best_Y = actual[0]
                    #check if we need to decrease the mutation_rate
                else:
                    Y = np.concatenate((Y, [actual]))

                X = np.vstack((X, x_sample))

            # retrain our bayesian model for improvement on new information.
            print("re-training surrogate model")
            checker.train_surrogate(X,Y)

            
            if len(best_x_close)>0:
                best_x_close= list(remove_duplicates(known_alternatives, best_x_close))
                if i>3: #store promising_alternatives_pool after nth learning epoch
                    promising_alternatives_pool.extend(best_x_close)
            # alternatives to be check after the last iteration, because the Bayesian model may be wiser then.
            BEST_alternatives_pool_arr.append(len(best_alternatives_pool))
            print("known alternatives size {}\n, X.shape {}, random_alternatives.shape: {}, new_max_epoch: {}, best_x_close: {}, best_alternatives_pool: {}, promising alternatives: {}".format(
                known_alternatives.shape[0],
                X.shape,
                random_alternatives.shape,
                new_max_epoch,
                len(best_x_close),
                len(best_alternatives_pool),
                len(promising_alternatives_pool)
            ))

        # LAST ITERATION
        # update the model
        if len(x_num_arr)==0 or ((new_max_epoch>=new_max_epoch_threshold or i==num_epochs) and i>min_epoch_threshold):
            print("Best alternative:",i,best_Y)
            
            threshold = .9
            promising_alternatives_pool.extend(best_alternatives_pool)
            # final check on the last iteration, with the wiser surrogate model.
            final_alternatives, _ = checker.check_promising_alternatives(
                promising_alternatives_pool,
                best_Y,
                best_alternatives_pool,
                threshold,
                target,
                positive_target)

            BEST_alternatives_pool_arr.append(len(final_alternatives))
            stoh_duration =  time.time() - stoh_start_time
  
            break
        #check if we need to create new alternatives
        # sample_too_small_bool = random_alternatives.shape[0]<random_alternatives_inital_size*random_sample_size_threshold
        # sample_too_small_bool = sample_too_small_bool or random_alternatives.shape[0]<20
        objective_zero_bool = objective_zero_threshold<=objective_zero
        objective_improvement_bool = improvement_zero_threshold<=improvement_zero
            
        #distance in the feature space from initial instance and the optimum solution
        num_changes = npu.distance(np.array(best_x),np.array(initial_instance),positive_target)-1
        changes_improvement_bool = current_num_changes>num_changes
        
        #check if we need to update the pool of random alternatives because we run out of neighbours and
        # the current pool is small OR
        # there have'nt been any improvement in the past K epochs OR
        # the objective function is stuck at 0 (no improvements on optimum)
        if len(best_x_close)==0 and (changes_improvement_bool or objective_zero_bool or objective_improvement_bool):
            current_num_changes = num_changes
            num_changes=changes_jitter+current_num_changes
 
            random_alternatives = generator.getRandom(random_sample_size, num_changes, positive_target)
            random_alternatives = remove_duplicates(known_alternatives, random_alternatives)

            objective_zero=0
            improvement_zero = 0    
            
    return initial_instance,final_alternatives,BEST_alternatives_pool_arr,stoh_duration,Y_epoch_mean,EST_epoch_mean
