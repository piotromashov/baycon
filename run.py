import numpy as np
import dex_bayesian_generator as bag_dsm
import dex_python_utilities as du

# perform the optimization process
local_path=''
positive_target = True
# possible output values that the DSM (or other plugin model) can have
output_values= ['low','medium','high']
# generates #first_sample of each possible output
first_sample = 3
# this is the pluggable model that we train to eXplain.
model = ' bm_lvl_4_normal_2_wo_links.dxi'
save_results = False
# level of neighbours, level of features to be changed from promissing counterfactual candidates.
neighbours_max_degree = 3
# 
target = 2

print("model:",model,'target:',target)
#generate starting alternatives and train the surrogate_model
X_string = du.generate_random_alternatives_DEX(first_sample,model) 
template_string = X_string[0] #starting laternative
print('Starting alternative:',template_string)
##run BAG DSM
print('running BAG-DSM')
template_numeric,final_alternatives,BEST_alternatives_pool_arr,stoh_duration,Y_epoch_mean,EST_epoch_mean = bag_dsm.run_generator(
    model, #DSM
    str(0),#index of initial instance
    template_string, #string representation of initial instance
    output_values, # low, medium, high
    target = target, # goal we want the achieve
    neighbours_max_degree=neighbours_max_degree,
    first_sample = first_sample, #generating initial instances once more? check it.
    positive_target = positive_target, #boolean flag on which direction to search the target (positive, negative)
    local_path=local_path,
    save_results=save_results,
    run='0' #the # of runs to average on which epoch the first solution is obtained
) 