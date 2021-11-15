import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import bayesian_generator as bcg_xai
import time_measurement
from common.DataAnalyzer import DataAnalyzer
from common.Target import Target

dataset_filename = "datasets/diabetes.csv"
target = Target(target_type="classification", target_value="tested_negative")

df = pd.read_csv(dataset_filename)
X = df.values[:, :-1]
Y = df.values[:, -1]

data_analyzer = DataAnalyzer(X, Y, target=target)

initial_instance_index = 0
initial_instance = X[initial_instance_index]
initial_prediction = Y[initial_instance_index]
model = RandomForestClassifier()  # pluggable model that we train to explain.
model.fit(X[initial_instance_index + 1:], Y[initial_instance_index + 1:])

instancesInfo = bcg_xai.run(initial_instance, initial_prediction, target, data_analyzer, model)

print(instancesInfo)

counterfactuals, scores = instancesInfo.achieved_score()
predictions = model.predict(counterfactuals)
output = {
    "initial_instance": initial_instance.tolist(),
    "initial_prediction": initial_prediction,
    "target_type": target.target_type(),
    "target_value": target.target_value(),
    "total_time": time_measurement.total_time,
    "time_to_first_solution": time_measurement.time_to_first_solution,
    "time_to_best_solution": time_measurement.time_to_best_solution,
    "counterfactuals": counterfactuals.tolist(),
    "predictions": predictions.tolist()
}
output_filename = "algorithm_output.json"
with open(output_filename, 'w') as outfile:
    json.dump(output, outfile)

