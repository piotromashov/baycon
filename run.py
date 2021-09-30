# import dex_bayesian_generator_commented as bag_dsm
import bayesian_generator as bag_dsm
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
import numpy_utils as npu
from DataConstraints import DataConstraints

# possible target values that the plugin model can have
# output_values= ['tested_negative', 'tested_positive']

dataset = fetch_openml(name='diabetes', version=1)

# transform data in the dataset from constant values into discrete ones using bins
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy='uniform')
discrete_dataset = discretizer.fit_transform(dataset.data)
#get information about the possible values for the features
data_constraints = DataConstraints(discrete_dataset)

# string representation of the initial instances.
random_alternatives = npu.generate_random_alternatives(discrete_dataset, n = 10) 

#generate starting alternatives and train the surrogate_model
initial_instance= discrete_dataset[len(discrete_dataset)-1]
print('Starting alternative:',initial_instance)
# this is the pluggable model that we train to eXplain.
model = RandomForestClassifier()
binary_target = [1 if t == "tested_positive" else 0 for t in dataset.target]
model.fit(discrete_dataset[0:-10], binary_target[0:-10])

##run BAG DSM
print('running BAG-DSM')
counterfactuals, time_to_first_solution = bag_dsm.run_generator(
    model, #DSM
    random_alternatives,
    data_constraints,
    initial_instance, #string representation of initial instance
    target = 1, # goal we want the achieve
)

print("initial instance: {}, output: {}".format(initial_instance, model.predict([initial_instance])))
outputs = model.predict(counterfactuals)

for counterfactual, output in zip(counterfactuals, outputs):
    distance = npu.distance(counterfactual, initial_instance)
    print("CF instance: {}, output: {}, distance: {}".format(counterfactual, output, distance))