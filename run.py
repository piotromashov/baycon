import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

import bayesian_generator as bag_dsm
from DataAnalyzer import DataAnalyzer

filename = "instances.csv"

dataset = fetch_openml(name='kc2', version=1)

# transform data in the dataset from constant values into discrete ones using bins
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy='uniform')
discrete_dataset = discretizer.fit_transform(dataset.data)
# get information about the possible values for the features
data_analyzer = DataAnalyzer(discrete_dataset)

# generate starting alternatives and train the surrogate_model
initial_instance = discrete_dataset[0]
print('Starting alternative:', initial_instance)
# pluggable model that we train to explain.
model = RandomForestClassifier()
binary_target = [1 if t == "yes" else 0 for t in dataset.target]
# binary_target = [1 if t == "tested_positive" else 0 for t in dataset.target]
model.fit(discrete_dataset[1:], binary_target[1:])

instancesInfo, time_to_first_solution = bag_dsm.run_generator(model, data_analyzer, initial_instance, target=1)

print("initial instance: {}, output: {}".format(initial_instance, model.predict([initial_instance])))
print("Generated counterfactuals {}".format(instancesInfo.achieved_target_count()))
for (score, count, instance) in instancesInfo.achieved_target_summary():
    print("Counterfactual with score {} ({}) {}".format("%.4f" % score, count, instance))

# save file
output = (initial_instance + instancesInfo.achieved_target())
pd.DataFrame(output).to_csv(filename, header=None, index=None)
