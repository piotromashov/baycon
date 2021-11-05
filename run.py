import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

import bayesian_generator as bcg_xai
import time_measurement
from DataAnalyzer import DataAnalyzer

dataset_filename = "datasets/diabetes.csv"
df = pd.read_csv(dataset_filename)
X = df.values[:, :-1]
Y = df.values[:, -1]

# transform data in the dataset from constant values into discrete ones using bins
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy='uniform')
X = discretizer.fit_transform(X)
# get information about the possible values for the features
data_analyzer = DataAnalyzer(X)

# generate starting alternatives and train the surrogate_model
initial_instance_index = 0
initial_instance = X[initial_instance_index]
initial_instance_prediction = Y[initial_instance_index]
opposite_prediction = 0 if initial_instance_prediction else 0
# Y = [1 if t == "yes" else 0 for t in Y]
Y = [1 if t == "tested_positive" else 0 for t in Y]
# pluggable model that we train to explain.
model = RandomForestClassifier()
model.fit(X[initial_instance_index + 1:], Y[initial_instance_index + 1:])

instancesInfo = bcg_xai.run(model, data_analyzer, initial_instance, target=opposite_prediction)

print(instancesInfo)

counterfactuals = instancesInfo.achieved_target().tolist()
output = {
    "initial_instance": initial_instance.tolist(),
    "counterfactuals": counterfactuals,
    "total_time": time_measurement.total_time,
    "time_to_first_solution": time_measurement.time_to_first_solution,
    "time_to_best_solution": time_measurement.time_to_best_solution
}
output_filename = "algorithm_output.json"
with open(output_filename, 'w') as outfile:
    json.dump(output, outfile)

