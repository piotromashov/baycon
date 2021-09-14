import subprocess
import os
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from warnings import catch_warnings,simplefilter
import pandas as pd
import acquisition_functions as acq_functions
import warnings
warnings.filterwarnings("ignore")
import time
import pickle
import dex_python_utilities as du


#distance metrics between two laternatives and a given target direction (positive or negative)
def distance(alternative,template,positive_target):
    if positive_target:
        dist = alternative-template
        dist[dist<0]=0
    else:
        dist =template-alternative
        dist[dist<0]=0
    return sum(dist)

#same as distance() applied over an array of alternatives
def distance_arr(alternative,template,positive_target):
    if positive_target:
        dist = alternative-template
        dist[dist<0]=0
    else:
        dist =template-alternative
        dist[dist<0]=0
    return np.sum(dist,axis=1)

#counts number of attribites that have lower (or higher for negative target) values than template
def difference_arr(alternative,template,positive_target):
    if positive_target:
        diff = alternative<template
    else:
        diff =template>alternative
    return np.sum(diff,axis=1)
    
#returns positions of attributes with differnet values than the template_numeric
def find_changes(alternative,template_numeric,positive_target):
    if positive_target:
        return np.argwhere(alternative>template_numeric)
    return np.argwhere(alternative<template_numeric)
# distance(best_x,template_numeric,positive_target)

#generates random alternatives for a given template
def generate_min_max_alternatives_from_template_2(n,known_alternatives,template,num_changes,
                                                  improvement=True,min_values=[],max_values=[]): 
    print("Generating alternatives from template.")
    if num_changes>len(template)/2:
        num_changes = num_changes//2
        
    min_change=1
    if num_changes==0:
        num_changes=1
    print("num_changes:",num_changes,'min_change:',min_change)
    
    if len(min_values)<1:
        min_values=2
        max_values=0
        print('here')
    
    n = 10000

    samples = np.random.rand(n,len(template)) #random matrix with n samples with values between 0 and 1

    max_change = max(max_values) #maximum allowed change for one attribute (e.g., from 0 to 3)
    
    ranges = (np.arange(0,100,100//(2*max_change+1))/100)[::-1][1:-1]
    new_values = []
    for n in range(max_change):
        new_values.append(n+1)
        new_values.append(-(n+1))
    samples[samples>ranges[0]]=new_values[0]  
    for k in range(len(ranges)-1):
        samples[(samples<=ranges[k]) & (samples>ranges[k+1])]=new_values[k+1]

    samples[(samples<=ranges[len(ranges)-1]) & (samples>=0)]=0

    samples = samples+template #increase each row with the template (i.e., increase template values by zero,one or two)
    
    #remove values bigger than max value
    while len(samples[samples>max_values])>0:
        samples[samples>max_values] = samples[samples>max_values]-1
    samples[samples<min_values]=0 

    samples = samples.astype(int)
    print(samples.shape)
    
    #increase atrute values that have lower values thetemplate
    
    #remove samples that are same as the template
    samples=samples[np.sum(samples!=template,axis=1)>0] 
    print(samples.shape)

    #remove samples that have more changes than max_num_changes ot less cnhanges than min_change
    d = distance_arr(samples,template,improvement)
    samples=samples[(d<=num_changes) & (d>=min_change)]
    
    print(samples.shape)
    #remove samples that already exist in known_alternatives
    last_idx = len(known_alternatives)
    samples = np.vstack((known_alternatives,samples))
    _,idx_arr = np.unique(samples,axis=0,return_index=True)
    idx_arr = idx_arr[idx_arr>last_idx]
    samples = samples[idx_arr]
    print(samples.shape)

    print("Done. Sample size:",samples.shape)
    return samples

def update_random_alternatives(random_sample_size,known_alternatives,template_numeric,num_changes,positive_target,min_values,max_values):
    return generate_min_max_alternatives_from_template_2(random_sample_size,known_alternatives,template_numeric,num_changes,positive_target,
                                                        min_values,max_values)

#calculate objective fuction for a list aletrnatives
def calculate_objective_all(model,attribute_values,alternatives,template_numeric,target,output_values,positive_target,run=0):
    cmd_eval_base = 'java -jar ./lib/DEXxSearch-1.2-dev.jar SIMPLE_EVALUATION models/'
    alternative_path =str(run)+'_template_alternatives.json'
    with open(alternative_path, 'w') as outfile:
#         json.dump(str(alternatives).replace(" ","").replace("'",'"'), outfile)
            json.dump(alternatives, outfile)

#     cmd = cmd_eval_base+model+" "+str(alternatives).replace(" ","").replace("'",'"')
    cmd = cmd_eval_base+model+" "+alternative_path

    s = subprocess.check_output(cmd,shell=True)
    str_value = json.loads(str(s)[2:-3])
#     print(str_value)
    evaluation_values = []
    for val in str_value:
        evaluation_values.append(int(output_values.index(val[0]['_'])))
#     print(evaluation_values)
    evaluation_values = np.array(evaluation_values)
    if positive_target:
        #target_eval = 1+(evaluation_values-target)
        target_eval = evaluation_values>=target
    else:
        #target_eval = 1+(target-evaluation_values)
        target_eval = evaluation_values<=target

    overall_num_changes = 2*len(template_numeric)
#     target_eval[target_eval>=2] = 1/overall_num_changes
#     target_eval[target_eval<0] = 0

#     print(target,'evaluation values:',evaluation_values)
    alternatives_numeric = du.alternatives_to_num(alternatives,attribute_values)
    alternatives_numeric = np.array(alternatives_numeric)
    template_numeric = np.array(template_numeric)
    
    num_changes = distance_arr(alternatives_numeric,template_numeric,positive_target)

    relative_similarity = 1-num_changes/overall_num_changes
    objective_value = relative_similarity*target_eval

#     print('relative_similarity:',relative_similarity,'target_eval:',
#           target_eval,'objective_value',objective_value)
    
    return objective_value


       
# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
    # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)

#returns mean values and standard deviation calculated over the predictions
#from each separate model from a given ensemble models
def get_ensemble_scores(ens_model,X):
    ens_predictions = []
    for est in range(len(ens_model.estimators_)): 
        ens_predictions.append(ens_model.estimators_[est].predict(X))
    ens_predictions = np.array(ens_predictions)
    mu = ens_predictions.mean(axis=0)
    return ens_predictions.mean(axis=0),ens_predictions.std(axis=0)

#returns an array of predictions from each separate model in a given ensemble model
def get_ensemble_predictions(ens_model,X):
    ens_predictions = []
    for est in range(len(ens_model.estimators_)): 
        ens_predictions.append(ens_model.estimators_[est].predict(X))
    return np.array(ens_predictions)


# returns scores caclulated with an acquisition function (see acqusition_functions.py)
def acquisition(X, X_candidates, model):
    yhat, _ = get_ensemble_scores(model,X)
    best = max(yhat)
    mu, std = get_ensemble_scores(model,X_candidates)
    score = acq_functions.EI(mu, std, best, epsilon=.001)
    return score

# optimize the acquisition function
from scipy.stats import norm
def opt_acquisition(X, y, model,X_candidates,attribute_values,template_string, top_n = 10):
    # calculate the acquisition function for each candidate
    scores = acquisition(X, X_candidates, model)
    # locate the index of the largest scores
    if top_n>len(scores)//2:
        top_n = len(scores)//2
    ix = np.argpartition(scores, -top_n)[-top_n:] #get top_n candidates
    best_alternative_numeric= X_candidates[ix]
    rand_idx =np.random.randint(0,len(X_candidates),top_n) #get_random_cadidates
    rand_alternative_numeric =  X_candidates[rand_idx]
    
    #remove candidates from the random candidates, as they will be available in X
    rm_idx = np.concatenate([rand_idx,ix])
    X_candidates = np.delete(X_candidates,rm_idx,axis=0)

    #join top-n candidates and n random cadindates and calculate their string representations
    alternatives_numeric = np.vstack([best_alternative_numeric,rand_alternative_numeric])
    alternatives_string = []
    k = sorted(template_string.keys())
    for a in alternatives_numeric:
        alternatives_string.append(du.atribute_values_to_string(a,k,attribute_values))
        
    return alternatives_string,alternatives_numeric,X_candidates


#updates columns at index indx with the given values val from a given matrix A, updates eac
#can be optimized. e.g., to avoid geenrating duplicates
def update_per_row(A, indx, val,num_elem=1):
    all_indx = indx[:,None] + np.arange(num_elem)
    A[np.arange(all_indx.shape[0])[:,None], all_indx] =val
    return A

#recoursive function
#probably can be optimized
#generates neighbours (i.e., similar solutions with same or samlled distance values than a given alternative x
#with respect to a templace
#max_degree tells hown many times the function will be called
#max_degree=1 (only x's neighbours)
#max_degree=2 (x's neighbours and neighbours of x's neighbours)
#max_degree=3 (x's neighbours, neighbours of x's neighbours, and neighbours of the neighbours of x's neighbours)
def generate_neighbours(x,template,positive_target,current_degree=1,x_close=[],max_degree=2):
    idx_arr = find_changes(x,template,positive_target)
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
                tmp_arr =update_per_row(tmp_arr,idx_arr_neg.flatten(),values)
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
def check_promising_alternatives(dex_model,model,alternatives,best_y,best_pool,threshold,template,template_string,attribute_values,target,output_values,positive_target,run):
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
        k = sorted(template_string.keys())
        alternatives_string = []
        for a in alternatives:
            alternatives_string.append(du.atribute_values_to_string(a,k,attribute_values))

        #check real objective value of all alternatives
        print(len(alternatives_string))
        Y_tmp=calculate_objective_all(dex_model,attribute_values,alternatives_string,template,target,output_values,positive_target,run)
        
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
                tmp_arr =update_per_row(tmp_arr,idx_arr_neg.flatten(),values)
                x_close.extend(tmp_arr)
        alternatives_tmp = x_close[len_before:]
        if len(x_close)==len_before or len(x_close)>top_n: #ro avoid memmory error
            break
    #check the real objective value of the neighbours
    if len(x_close)>0:
        x_close =  remove_duplicates(x_close,alternatives)
        if len(x_close)>0:  
            print(len(x_close))
            alternatives_string = []
            k = sorted(template_string.keys())

            for a in x_close:
                alternatives_string.append(du.atribute_values_to_string(a,k,attribute_values))

            Y_tmp=calculate_objective_all(dex_model,attribute_values,alternatives_string,template,target,output_values,positive_target,run)
            
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
    min_values = np.zeros((len(attribute_values)))
    max_values = []
    for av in attribute_values:
        max_values.append(len(av)-1)
    return min_values,np.array(max_values)

def acquisition_random(X_candidates,attribute_values,template_string, top_n = 20):
    
    rand_idx =np.random.randint(0,len(X_candidates),top_n) #get_random_cadidates
    rand_alternative_numeric =  X_candidates[rand_idx]
    
    #remove candidates from the random candidates, as they will be available in X
    X_candidates = np.delete(X_candidates,rand_idx,axis=0)

    #join top-n candidates and n random cadindates and calculate their string representations
    alternatives_numeric = rand_alternative_numeric
    alternatives_string = []
    k = sorted(template_string.keys())
    for a in alternatives_numeric:
        alternatives_string.append(du.atribute_values_to_string(a,k,attribute_values))
        
    return alternatives_string,alternatives_numeric,X_candidates


def run_generator(model,starting_alternative,template_string,output_values,target = 2, neighbours_max_degree=3,
                  first_sample = 3,positive_target = True, local_path = "F:/DS/",save_results=True,run=0):
    
    print("model:",model,'target:',target,'positive_target:',positive_target)
    
    #get information about the model, which features we have and their possible values
    attributes,attribute_category,attribute_values  = du.get_model_description(model)
    
    #random instances for the bayesian model
    random_sample_size=10000
    #overall number of epochs to run the algorithm
    num_epochs = 150 
    #generate new samples when the size is lower than x% of the initial size
    random_sample_size_threshold = .1 
    #generate new samples when the average objective values has not incrased for x epochs
    objective_zero_threshold = 3
    #improvement
    improvement_zero_threshold = 50   
    current_num_changes=0
    new_max_epoch_threshold = 50
    min_epoch_threshold = 100
                           
    print()
    print('-----------')
    model  =" "+model
    attributes,attribute_category,attribute_values  = du.get_model_description(model)
    surrogate_model = RandomForestRegressor(1000,n_jobs=4)
    
    #generate starting dataset and train the surrogate_model
    X_string = du.generate_random_alternatives_DEX(first_sample,model) 
    X = du.alternatives_to_num(X_string,attribute_values)
    template_numeric =du.atribute_values_to_num(template_string,attribute_values)
    Y=calculate_objective_all(model,attribute_values,X_string,template_numeric,target,output_values,positive_target,run)
    surrogate_model.fit(X,Y)
    known_alternatives = X.copy() #known_alternatives to avoid dulicates

    #for log
    print('model:',model,'first_sample:',first_sample,'target:',target,'template:',starting_alternative,'positive_target:',positive_target)

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
        best_x_close = generate_neighbours(best_x,template_numeric,positive_target,1,[],neighbours_max_degree)

    current_num_changes = distance(np.array(best_x),np.array(template_numeric),positive_target)-1
    
    i=0
    objective_zero = 0 #counter - number of epochs without objective improvement
    improvement_zero=0 #counter - number of epochs improvement being zero
    secondary_counter=0 #counter - number of checks of the secondary pool
    pool_num = []
    pool_string = []
    
    EST_epoch_mean = []
    Y_epoch_mean= []
    BEST_alternatives_pool_arr = []
    stoh_start_time = time.time()
    random_alternatives = np.array([[1],[1]])
    random_alternatives_inital_size = 1
    
    neighborhood_jitter= .75
    changes_jitter=8
    print('neighborhood_jitter',neighborhood_jitter)
    print('neighbours_max_degree',neighbours_max_degree)
    
    best_alternatives_pool = []
    best_alternatives_pool.append(list(best_x))
    
    promising_alternatives_pool = []
    promising_alternatives_pool.extend(best_x_close)
    new_max_epoch = 0
    x_string_arr = []
    
    
    min_values, max_values = generate_minMax_constrais(attribute_values)
    
    while i < num_epochs:
        print('----')
        print(i,'epoch')
        #helper variables
        objective_zero = objective_zero+1
        improvement_zero = improvement_zero+1
        i=i+1
        new_max_epoch = new_max_epoch+1
        
        #check if we have close neighbours to be checked
        if len(best_x_close)>0:
            print('neighbours to be checked:',len(best_x_close))
            x_string_arr,x_num_arr,_ = opt_acquisition(X, Y, surrogate_model,np.array(best_x_close),attribute_values, template_string)
            print('predicting...')
            est_arr = surrogate_model.predict(x_num_arr)
            best_x_close = []
            best_x_close_string = []
        else:
            if len(random_alternatives)<10:
                num_changes=changes_jitter+current_num_changes
                random_alternatives = generate_min_max_alternatives_from_template_2(random_sample_size,known_alternatives,template_numeric,num_changes,positive_target,
                                                                                   min_values,max_values) 
                random_alternatives_inital_size = random_alternatives.shape[0]
            if len(random_alternatives)>0:
                x_string_arr,x_num_arr,random_alternatives = opt_acquisition(X, Y, surrogate_model,random_alternatives,attribute_values,template_string)
                print('predicting...')
                est_arr = surrogate_model.predict(x_num_arr)
    
        if len(x_string_arr)!=0:
            print('calculating objective...')
            actual_arr = []
            actual_arr = calculate_objective_all(model,attribute_values,x_string_arr,template_numeric,target,output_values,positive_target,run)

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
                est = est_arr[k]
                x_sample = x_num_arr[k]
                x_string = x_string_arr[k]

                if actual>0.01:
                    objective_zero=0 #restart counter
                   # improvement_weight = len(X)/10 #oversample improvement instance using improvement_weight
                improvement_weight = 1

                if actual>=(current_best_Y*neighborhood_jitter) and actual>0.01:
                    best_x_close_tmp = generate_neighbours(x_sample,template_numeric,positive_target,1,[],neighbours_max_degree)
                    best_x_close.extend(best_x_close_tmp)

                if actual>=current_best_Y and actual>0.01: #new estimated optimum found
                    best_x = x_sample
                    if actual>current_best_Y: #restart best aelernatives
                        best_alternatives_pool=[]                            
                        BEST_alternatives_pool_arr = list(np.repeat(0,len(BEST_alternatives_pool_arr)))
                    if list(best_x) not in best_alternatives_pool:
                        best_alternatives_pool.append(list(best_x))
                    new_max_epoch = 0

                    improvement_zero=1 #restart counter
                    x_sample = np.repeat([x_sample],improvement_weight,axis=0) #oversample
                    actual = np.repeat(actual,improvement_weight) #oversample
                    Y = np.concatenate((Y, actual))
                    current_best_Y = actual[0]
                    #check if we need to decrease the mutation_rate
                else:
                    Y = np.concatenate((Y, [actual]))

                X = np.vstack((X, x_sample))
                X_string.append(x_string)

            surrogate_model.fit(X,Y)

            if len(best_x_close)>0:
                best_x_close= remove_duplicates(best_x_close,known_alternatives)
                if i>3: #store promising_alternatives_pool after nth learning epoch
                    promising_alternatives_pool.extend(best_x_close)
            BEST_alternatives_pool_arr.append(len(best_alternatives_pool))
            print(known_alternatives.shape,X.shape,random_alternatives.shape,new_max_epoch,len(best_x_close),len(best_alternatives_pool),len(promising_alternatives_pool))

        # update the model
        if len(x_string_arr)==0 or ((new_max_epoch>=new_max_epoch_threshold or i==num_epochs) and i>min_epoch_threshold):
    
            print("Best alternative:",i,current_best_Y)
            m = X[np.argmax(Y)]
            k = sorted(template_string.keys())
            ms = du.atribute_values_to_string(m,k,attribute_values)

            threshold = .9
            promising_alternatives_pool.extend(best_alternatives_pool)
            final_alternatives,final_y = check_promising_alternatives(model,surrogate_model,
                    promising_alternatives_pool,current_best_Y,best_alternatives_pool,
                    threshold,template_numeric,template_string,
                    attribute_values,target,output_values,
                    positive_target,run)

            BEST_alternatives_pool_arr.append(len(final_alternatives))
            stoh_duration =  time.time() - stoh_start_time
            

            print("stochastic",stoh_duration,calculate_objective_all(model,attribute_values,[ms],template_numeric,target,output_values,positive_target))
            print("template",calculate_objective_all(model,attribute_values,[template_string],template_numeric,target,output_values,positive_target))
            path = local_path+"ML_Models/"+model+"_"+str(target)+"_"+str(int(positive_target))+"_"+starting_alternative+"_RF.pickle"
            if (save_results):
                print('Saving ML model to ',path)
                pickle.dump(surrogate_model, open(path, 'wb'))
                print('Done')

                path = local_path+"Alternatives/"+model+"_"+str(target)+"_"+str(int(positive_target))+"_"+starting_alternative+"_alternatives.pickle"
                print('Saving alternatives to ',path)
                pickle.dump([known_alternatives, promising_alternatives_pool,final_alternatives], open(path, 'wb'))
                print('Done')
  
            break
        #check if we need to create new alternatives
        sample_too_small_bool = random_alternatives.shape[0]<random_alternatives_inital_size*random_sample_size_threshold
        sample_too_small_bool = sample_too_small_bool or random_alternatives.shape[0]<20
        objective_zero_bool = objective_zero_threshold<=objective_zero
        objective_improvement_bool = improvement_zero_threshold<=improvement_zero
          #max_mutation_rate_NOT_reached_bool = mutation_rate<max_mutation_rate
        num_changes = distance(np.array(best_x),np.array(template_numeric),positive_target)-1
        changes_improvement_bool = current_num_changes>num_changes
    
        if len(best_x_close)==0 and (changes_improvement_bool or objective_zero_bool or objective_improvement_bool):
            current_num_changes = num_changes
            num_changes=changes_jitter+current_num_changes
            random_alternatives = update_random_alternatives(random_sample_size,known_alternatives,template_numeric,num_changes,positive_target,
                                                            min_values,max_values)
            random_alternatives_inital_size = random_alternatives.shape[0]
            objective_zero=0
            improvement_zero = 0    
            
            
    return template_numeric,final_alternatives,BEST_alternatives_pool_arr,stoh_duration,Y_epoch_mean,EST_epoch_mean


def run_random_generator(model,starting_alternative,template_string,output_values,meta_data,target = 2, neighbours_max_degree=3,first_sample = 3,positive_target = True, local_path = "F:/DS/",save_results=True,run=0,max_duration=0):
      
    print("model:",model,'target:',target,'positive_target:',positive_target)
        
    attributes,attribute_category,attribute_values  = du.get_model_description(model)
    random_sample_size=10000
    num_epochs = 150 #overall number of epochs to run the algorithm
    #generate new samples when the size is lower than x% of the initial size
    random_sample_size_threshold = .1 
    #generate new samples when the average objective values has not incrased for x epochs
    objective_zero_threshold = 3
    #improvement
    improvement_zero_threshold = 50   
    current_num_changes=0
    new_max_epoch_threshold = 50
    min_epoch_threshold = 100
                           
    print()
    print('-----------')
    model  =" "+model
    attributes,attribute_category,attribute_values  = du.get_model_description(model)
    
    #generate starting dataset and train the surrogate_model
    X_string = du.generate_random_alternatives_DEX(first_sample,model) 
    X = du.alternatives_to_num(X_string,attribute_values)
    template_numeric =du.atribute_values_to_num(template_string,attribute_values)
    Y=calculate_objective_all(model,attribute_values,X_string,template_numeric,target,output_values,positive_target,run)
    known_alternatives = X.copy() #known_alternatives to avoid dulicates

    #for log
    print('model:',model,'first_sample:',first_sample,'target:',target,'template:',starting_alternative,'positive_target:',positive_target)

    #oversample current best if it carries some information
    best_x = X[np.argmax(Y)]
    best_x_close = []
    best_x_close_string = []
    current_best_Y = max(Y)
    if current_best_Y>0:
        current_best = X[np.argmax(Y)]
        actual = current_best_Y
        Y = np.concatenate((Y, [actual]))
        
    best_alternatives_pool = []
    best_alternatives_pool.append(list(best_x))

    current_num_changes = distance(np.array(best_x),np.array(template_numeric),positive_target)
    
    objective_zero = 0 #counter - number of epochs without objective improvement
    improvement_zero=0 #counter - number of epochs improvement being zero
    secondary_counter=0 #counter - number of checks of the secondary pool
    pool_num = []
    pool_string = []
    
    Y_epoch_mean= []
    BEST_alternatives_pool_arr = []
    stoh_start_time = time.time()
    random_alternatives = np.array([[1],[1]])
    random_alternatives_inital_size = 1
    
    

    promising_alternatives_pool = []
    new_max_epoch = 0
    
    
    min_values, max_values = generate_minMax_constrais(attribute_values)
    if max_duration==0:
        max_duration =  meta_data[meta_data.template_idx==starting_alternative].stoh_duration.values[0]
    i=0
    stoh_duration = 0
    x_string_arr = []
    while True:
        print('----')
        print('duration:',stoh_duration,'out of',max_duration,'seconds')
        print(i,'epoch')
        #helper variables
        objective_zero = objective_zero+1
        improvement_zero = improvement_zero+1
        i=i+1
        new_max_epoch = new_max_epoch+1
        
        if len(random_alternatives)<10:
            num_changes=current_num_changes
            random_alternatives = generate_min_max_alternatives_from_template_2(random_sample_size,known_alternatives,template_numeric,num_changes,positive_target,min_values,max_values) 
            random_alternatives_inital_size = random_alternatives.shape[0]
        if len(random_alternatives)>0:
            x_string_arr,x_num_arr,random_alternatives = acquisition_random(random_alternatives,
                                                                                attribute_values,
                                                                                template_string)
    
        if len(x_string_arr)!=0:
            print('calculating objective...')
            actual_arr = []
            actual_arr = calculate_objective_all(model,attribute_values,x_string_arr,template_numeric,target,output_values,positive_target,run)

            #for log - save average estimated and real objective values 
            if len(actual_arr)>10:
                Y_epoch_mean.extend([np.mean(actual_arr[:10]),np.mean(actual_arr[10:])])
            else:
                Y_epoch_mean.extend([np.mean(actual_arr),-1])

            #add to known_alternatives
            known_alternatives = np.vstack([known_alternatives,x_num_arr])

            print('actual:',np.round(actual_arr,3),np.round(max(Y),3))

            # add the data to the dataset for the surogate model
            for k in range(len(actual_arr)):
                actual = actual_arr[k]
                x_sample = x_num_arr[k]
                x_string = x_string_arr[k]

                improvement_weight = 1

                if actual>=current_best_Y and actual>0.01: #new estimated optimum found
                    best_x = x_sample
                    
                    if actual>current_best_Y:
                        best_alternatives_pool=[]
                        BEST_alternatives_pool_arr = list(np.repeat(0,len(BEST_alternatives_pool_arr)))


                    if list(best_x) not in best_alternatives_pool:
                        best_alternatives_pool.append(list(best_x))
                        
                    new_max_epoch = 0

                    improvement_zero=1 #restart counter

                    Y = np.concatenate((Y, [actual]))
                    current_best_Y = actual
                    #check if we need to decrease the mutation_rate
                else:
                    Y = np.concatenate((Y, [actual]))

                X = np.vstack((X, x_sample))
                X_string.append(x_string)
                
            BEST_alternatives_pool_arr.append(len(best_alternatives_pool))
            print(known_alternatives.shape,X.shape,random_alternatives.shape,new_max_epoch,len(best_x_close),len(best_alternatives_pool),len(promising_alternatives_pool))


       # update the model
        stoh_duration =  time.time() - stoh_start_time
        if stoh_duration>=max_duration:
    
            print("Best alternative:",i,current_best_Y)
            m = X[np.argmax(Y)]
            k = sorted(template_string.keys())
            ms = du.atribute_values_to_string(m,k,attribute_values)

            threshold = .9
            final_alternatives = best_alternatives_pool

            BEST_alternatives_pool_arr.append(len(final_alternatives))
            

            print("stochastic",stoh_duration,calculate_objective_all(model,attribute_values,[ms],template_numeric,target,output_values,positive_target))
            print("template",calculate_objective_all(model,attribute_values,[template_string],template_numeric,target,output_values,positive_target))

            if (save_results):

                path = local_path+"Alternatives/random_"+model+"_"+str(target)+"_"+str(int(positive_target))+"_"+starting_alternative+"_alternatives.pickle"
                print('Saving alternatives to ',path)
                pickle.dump([known_alternatives, promising_alternatives_pool,final_alternatives], open(path, 'wb'))
                print('Done')
  
            break
        #check if we need to create new alternatives
        sample_too_small_bool = random_alternatives.shape[0]<random_alternatives_inital_size*random_sample_size_threshold
        sample_too_small_bool = sample_too_small_bool or random_alternatives.shape[0]<20
        objective_zero_bool = objective_zero_threshold<=objective_zero
        objective_improvement_bool = improvement_zero_threshold<=improvement_zero
          #max_mutation_rate_NOT_reached_bool = mutation_rate<max_mutation_rate
        num_changes = distance(np.array(best_x),np.array(template_numeric),positive_target)
        changes_improvement_bool = current_num_changes>num_changes

    
        if len(best_x_close)==0 and (changes_improvement_bool or objective_zero_bool or objective_improvement_bool):
            current_num_changes = num_changes
            random_alternatives = update_random_alternatives(random_sample_size,known_alternatives,template_numeric,num_changes,positive_target,
                                                            min_values,max_values)
            random_alternatives_inital_size = random_alternatives.shape[0]
            objective_zero=0
            improvement_zero = 0    
    

    return template_numeric,final_alternatives,BEST_alternatives_pool_arr,stoh_duration,Y_epoch_mean
