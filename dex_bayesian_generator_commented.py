import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import acquisition_functions as acq_functions
import warnings
import numpy_utils as npu
warnings.filterwarnings("ignore")
import time
from InstancesGenerator import InstancesGenerator

#generates random alternatives for a given template
def generate_min_max_alternatives_from_template(n,known_alternatives,template,num_changes,
                                                  improvement=True,min_values=[],max_values=[]): 
    print("Generating alternatives from template.")
    features_amount = len(template)
    if num_changes>features_amount/2:
        num_changes = num_changes//2
        
    min_change=1
    if not num_changes:
        num_changes=min_change
    print("num_changes:",num_changes,'min_change:',min_change)

    samples = np.random.rand(n,len(template)) #random matrix with n samples with values between 0 and 1
    
    #maximum and minimum allowed change for one attribute (e.g., from 0 to 3)
    max_value = max(max_values) 
    min_value = -max_value

    random_matrix = np.random.randint(min_value,max_value,(n,features_amount))

    samples = random_matrix+template #increase each row with the template (i.e., increase template values by zero,one or two)
    
    #remove values bigger than max value
    while len(samples[samples>max_values])>0:
        samples[samples>max_values] = samples[samples>max_values]-1
    samples[samples<min_values]=0 

    samples = samples.astype(int)
    
    #increase atrute values that have lower values thetemplate
    
    #remove samples that are same as the template
    samples=samples[np.sum(samples!=template,axis=1)>0] 

    #remove samples that have more changes than max_num_changes ot less cnhanges than min_change
    d = npu.distance_arr(samples,template,improvement)
    samples=samples[(d<=num_changes) & (d>=min_change)]
    
    #remove samples that already exist in known_alternatives
    last_idx = len(known_alternatives)
    samples = np.vstack((known_alternatives,samples))
    _,idx_arr = np.unique(samples,axis=0,return_index=True)
    idx_arr = idx_arr[idx_arr>last_idx]
    samples = samples[idx_arr]

    print("Done. Sample size:",samples.shape)
    return samples

def update_random_alternatives(random_sample_size,known_alternatives,template_numeric,num_changes,positive_target,min_values,max_values):
    return generate_min_max_alternatives_from_template(
        random_sample_size,
        known_alternatives,
        template_numeric,
        num_changes,
        positive_target,
        min_values,
        max_values
    )

#calculate objective fuction for a list aletrnatives
def calculate_objective_all(model,alternatives,template_numeric,target,positive_target):
    #get model prediction on those values
    Y = model.predict(alternatives)
    Y = np.array(Y)

    #TODO: what is this 2 multiplying?
    overall_num_changes = 2*len(template_numeric)

    # here should go the cost of attribute changes and their weights
    num_changes = npu.distance_arr(alternatives, template_numeric, positive_target)

    # closeness to feature space of the potential counterfactual to the initial instance.
    relative_similarity = 1-num_changes/overall_num_changes

    # check if we are moving towards the target or not.
    #if we are not moving towards the target, this is weighted as 0
    target_achieved = Y==target
    objective_value = relative_similarity*target_achieved    
    return objective_value

#returns mean values and standard deviation calculated over the predictions
#from each separate model from a given ensemble models
def get_ensemble_scores(ens_model,X):
    ens_predictions = []
    for est in range(len(ens_model.estimators_)): 
        #TODO: what are those estimators?
        ens_predictions.append(ens_model.estimators_[est].predict(X))
    ens_predictions = np.array(ens_predictions)

    mu = ens_predictions.mean(axis=0)
    std = ens_predictions.std(axis=0)
    return mu, std

#returns an array of predictions from each separate model in a given ensemble model
def get_ensemble_predictions(ens_model,X):
    ens_predictions = []
    for est in range(len(ens_model.estimators_)): 
        ens_predictions.append(ens_model.estimators_[est].predict(X))
    return np.array(ens_predictions)

# returns scores caclulated with an acquisition function (see acqusition_functions.py)
def acquisition(model, X, X_candidates):
    mu, _ = get_ensemble_scores(model,X)
    best_mu = max(mu)
    mu, std = get_ensemble_scores(model,X_candidates)
    score = acq_functions.EI(mu, std, best_mu, epsilon=.001)
    return score

# optimize the acquisition function
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

#recoursive function
#probably can be optimized
#generates neighbours (i.e., similar solutions with same or samlled distance values than a given alternative x
#with respect to a templace
#max_degree tells hown many times the function will be called
#max_degree=1 (only x's neighbours)
#max_degree=2 (x's neighbours and neighbours of x's neighbours)
#max_degree=3 (x's neighbours, neighbours of x's neighbours, and neighbours of the neighbours of x's neighbours)
def generate_neighbours(x,template,positive_target,current_degree=1,x_close=[],max_degree=2):
    idx_arr = npu.find_changes(x,template,positive_target)
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
                idx_arr_neg  = np.argwhere(tmp_x<template)
                values = tmp_x[idx_arr_neg]+1
            else:
                idx_arr_neg  = np.argwhere(tmp_x>template)
                values = tmp_x[idx_arr_neg]-1
            if len(idx_arr_neg)>0:
                tmp_arr = np.repeat([tmp_x],len(idx_arr_neg),axis=0)
                tmp_arr =npu.update_per_row(tmp_arr,idx_arr_neg.flatten(),values)
                x_close.extend(tmp_arr)
            
            generate_neighbours(tmp_x,template,positive_target,
                                current_degree,x_close,max_degree) 
    #remove duplicates
    if len(x_close)>0:
        if len(x_close)>1:
            x_close = np.unique(x_close,axis=0)
        return list(x_close)
    return []
        
#removes from "samples" alternatives that already exist in known_alternatives 
def remove_duplicates(samples,known_alternatives):
    last_idx = len(known_alternatives)-1
    samples = np.vstack((known_alternatives,samples))
    _,idx_arr = np.unique(samples,axis=0,return_index=True) #remove duplicate samples
    idx_arr = idx_arr[idx_arr>last_idx]
    samples = samples[idx_arr]
    return list(samples)

#check promising_alternatives_pool 
def check_promising_alternatives(dex_model,model,alternatives,best_y,best_pool,threshold,template,target,positive_target):
    print('checking promising_alternatives')
    top_n = 1000
    if len(alternatives)==0:
        alternatives = np.array(best_pool)[:top_n]
    else:
        #predict the optimization function with the surogate model
        if np.array(alternatives).ndim==1:
            pred = model.predict(np.array(alternatives).reshape(1, -1))
        else:
            pred = model.predict(alternatives)
        alternatives = np.array(alternatives)
        print(alternatives.shape)

        #get n-top ranked alternatives
   
        alternatives = alternatives[pred>=best_y*threshold]
        pred = pred[pred>=best_y*threshold]
        print(alternatives.shape)
        if len(alternatives)>top_n:
            ix = np.argpartition(pred, -top_n)[-top_n:] #get top_n candidates
            alternatives= alternatives[ix]

    print(alternatives.shape)
    Y = np.repeat(best_y,len(best_pool))
    #for each top-ranked alternative
    #check real objective value of the alternative and its neighbours
    if len(alternatives)>0:
        #check real objective value of all alternatives
        Y_tmp=calculate_objective_all(dex_model,alternatives,template,target,positive_target)
        
        alternatives = np.vstack((alternatives,best_pool))
        Y = np.concatenate((Y_tmp,Y))
    else:
         alternatives = np.array(best_pool)

    alternatives =  alternatives[[Y>=best_y]]
    Y = Y[[Y>=best_y]]
    print('candidates best_y',len(Y),alternatives.shape)

    #generate neighbours
    x_close = []
    alternatives_tmp = alternatives
    for i in range(10):
        len_before = len(x_close) 
        for tmp_x in alternatives_tmp:
            if positive_target: #this updaee would not change the distance value between the template and the alternative
                idx_arr_neg  = np.argwhere(tmp_x<template)
                values = tmp_x[idx_arr_neg]+1
            else:
                idx_arr_neg  = np.argwhere(tmp_x>template)
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
        x_close =  remove_duplicates(x_close,alternatives)
        
        if len(x_close)>0:  
            Y_tmp=calculate_objective_all(dex_model,x_close,template,target,positive_target)
            alternatives = np.vstack((alternatives,x_close))
            Y = np.concatenate((Y,Y_tmp))
            alternatives = alternatives[Y>=best_y]
            Y = Y[Y>=best_y]
    alternatives = np.unique(alternatives,axis=0)
    print("unique Y:",alternatives.shape)
    print('best_y',len(Y))
    return alternatives,Y


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

    # objective function, for now we work with this single number (single objective optimization)
    Y=calculate_objective_all(
        model,
        X,
        initial_instance,
        target,
        positive_target
    )
    
    surrogate_model.fit(X,Y)
    known_alternatives = X.copy() #known_alternatives to avoid duplicates

    #for log
    print('model:',model,'first_sample:',first_sample,'target:',target,'template:',initial_instance,'positive_target:',positive_target)

    #oversample current best if it carries some information
    best_x = X[np.argmax(Y)]
    best_x_close = []
    best_x_close_string = []
    current_best_Y = max(Y)
    if current_best_Y>0:
        current_best = X[np.argmax(Y)]
        current_best = np.repeat([current_best],10,axis=0)
        X = np.vstack((X, current_best))
        actual = current_best_Y
        actual = np.repeat(actual,10)
        Y = np.concatenate((Y, actual))
        
        #save best_x neighbours to be checked in the next iteration
        best_x = current_best[0]                
        best_x_close = generate_neighbours(best_x,initial_instance,positive_target,1,[],neighbours_max_degree)

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
    random_alternatives_inital_size = 1
    
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
    
    min_values, max_values = generate_minMax_constrais(feature_values)
    generator = InstancesGenerator(initial_instance, min_values, max_values)
    
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
            #logging purposes on the prediction
            est_arr = surrogate_model.predict(x_num_arr)
            best_x_close = []
            best_x_close_string = []
        else:
            #go with random counterfactuals
            if len(random_alternatives)<10:
                # how many features we allow to change
                num_changes=changes_jitter+current_num_changes
                random_alternatives = generator.getRandom(random_sample_size, num_changes, positive_target)
                random_alternatives = generator.update(known_alternatives, random_alternatives)
                
                #number of unique generated instances
                random_alternatives_inital_size = random_alternatives.shape[0]
            if len(random_alternatives)>0:
                x_num_arr,random_alternatives = opt_acquisition(surrogate_model, X,random_alternatives)
                print('predicting...')
                #logging purposes on the prediction
                est_arr = surrogate_model.predict(x_num_arr)
    
        if len(x_num_arr)!=0:
            print('calculating objective...')
            actual_arr = []
            actual_arr = calculate_objective_all(model,x_num_arr,initial_instance,target,positive_target)

            #for log - save average estimated and real objective values 
            if len(est_arr)>10:
                EST_epoch_mean.extend([np.mean(est_arr[:10]),np.mean(est_arr[10:])]) #first 5 are infromative samples second 5 are random samples
                Y_epoch_mean.extend([np.mean(actual_arr[:10]),np.mean(actual_arr[10:])])
            else:
                EST_epoch_mean.extend([np.mean(est_arr),-1]) #first 5 are infromative samples second 5 are random samples
                Y_epoch_mean.extend([np.mean(actual_arr),-1])

            #add to known_alternatives
            known_alternatives = np.vstack([known_alternatives,x_num_arr])

            print('estimated:',np.round(est_arr,3))
            print('actual:',np.round(actual_arr,3),np.round(max(Y),3))

            # add the data to the dataset for the surogate model
            for k in range(len(actual_arr)):
                actual = actual_arr[k]
                x_sample = x_num_arr[k]

                if actual>0.01:
                    objective_zero=0 #restart counter
                   # improvement_weight = len(X)/10 #oversample improvement instance using improvement_weight
                improvement_weight = 1

                # check if close to current best, then it's candidate to keep exploring
                if actual>=(current_best_Y*neighborhood_jitter) and actual>0.01:
                    best_x_close_tmp = generate_neighbours(x_sample,initial_instance,positive_target,1,[],neighbours_max_degree)
                    best_x_close.extend(best_x_close_tmp)

                if actual>=current_best_Y and actual>0.01: #new estimated optimum found
                    best_x = x_sample
                    if actual>current_best_Y: #restart best alternatives
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
                    current_best_Y = actual[0]
                    #check if we need to decrease the mutation_rate
                else:
                    Y = np.concatenate((Y, [actual]))

                X = np.vstack((X, x_sample))

            # retrain our bayesian model for improvement on new information.
            surrogate_model.fit(X,Y)

            
            if len(best_x_close)>0:
                best_x_close= remove_duplicates(best_x_close,known_alternatives)
                if i>3: #store promising_alternatives_pool after nth learning epoch
                    promising_alternatives_pool.extend(best_x_close)
            # alternatives to be check after the last iteration, because the Bayesian model may be wiser then.
            BEST_alternatives_pool_arr.append(len(best_alternatives_pool))
            print("known alternatives shape {}, X.shape {}, random_alternatives.shape: {}, new_max_epoch: {}, best_x_close: {}, best_alternatives_pool: {}, promising alternatives: {}".format(
                known_alternatives.shape,
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
            print("Best alternative:",i,current_best_Y)
            
            threshold = .9
            promising_alternatives_pool.extend(best_alternatives_pool)
            # final check on the last iteration, with the wiser surrogate model.
            final_alternatives, _ = check_promising_alternatives(model,surrogate_model,
                    promising_alternatives_pool,current_best_Y,best_alternatives_pool,
                    threshold,initial_instance,target,
                    positive_target)

            BEST_alternatives_pool_arr.append(len(final_alternatives))
            stoh_duration =  time.time() - stoh_start_time
  
            break
        #check if we need to create new alternatives
        sample_too_small_bool = random_alternatives.shape[0]<random_alternatives_inital_size*random_sample_size_threshold
        sample_too_small_bool = sample_too_small_bool or random_alternatives.shape[0]<20
        objective_zero_bool = objective_zero_threshold<=objective_zero
        objective_improvement_bool = improvement_zero_threshold<=improvement_zero
          #max_mutation_rate_NOT_reached_bool = mutation_rate<max_mutation_rate
            
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
            random_alternatives = generator.update(known_alternatives, random_alternatives)

            random_alternatives_inital_size = random_alternatives.shape[0]
            objective_zero=0
            improvement_zero = 0    
            
            
    return initial_instance,final_alternatives,BEST_alternatives_pool_arr,stoh_duration,Y_epoch_mean,EST_epoch_mean
