import bayesian_generator as bag_dsm
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
import numpy_utils as npu
from DataConstraints import DataConstraints
dataset = fetch_openml(name='diabetes', version=1)

# transform data in the dataset from constant values into discrete ones using bins
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy='uniform')
discrete_dataset = discretizer.fit_transform(dataset.data)
# get information about the possible values for the features
data_constraints = DataConstraints(discrete_dataset)

# generate starting alternatives and train the surrogate_model
initial_instance = discrete_dataset[len(discrete_dataset) - 1]
print('Starting alternative:', initial_instance)
# this is the pluggable model that we train to eXplain.
model = RandomForestClassifier()
binary_target = [1 if t == "tested_positive" else 0 for t in dataset.target]
model.fit(discrete_dataset[0:-10], binary_target[0:-10])

# run BAG DSM
print('running BAG-DSM')
instancesInfo, time_to_first_solution = bag_dsm.run_generator(
    model,  # DSM
    data_constraints,
    initial_instance,  # string representation of initial instance
    target=1,  # goal we want the achieve
)

print("initial instance: {}, output: {}".format(initial_instance, model.predict([initial_instance])))
print("Generated counterfactuals {}".format(instancesInfo.achieved_target_count()))
for (distance, count, score) in instancesInfo.achieved_target_summary():
    print("Instances with distance {} are {}, score: {}".format(distance, count, score))
