import json

import numpy as np
import pandas as pd

import bcgxai.bayesian_generator as bcg_xai
import bcgxai.time_measurement as time_measurement
from common.DataAnalyzer import DataAnalyzer
from common.Target import Target

# dataset_filename = "datasets/diabetes.csv"
# target = Target(target_type="classification", target_feature="class", target_value="tested_negative")
# initial_instance_index = 0

# dataset_filename = "datasets/kc2.csv"
# target = Target(target_type="classification", target_feature="problems", target_value="no")
# initial_instance_index = 4

# dataset_filename = "datasets/pd_speech_features.csv"
# target = Target(target_type="classification", target_feature="class", target_value=0)
# initial_instance_index = 0

dataset_filename = "datasets/house_sales.csv"
target = Target(target_type="regression", target_feature="price", target_value="decrease")
initial_instance_index = 0
cat_features = ["waterfront", "date_year"]

df = pd.read_csv(dataset_filename)
data_analyzer = DataAnalyzer(df, target=target)
X, Y = data_analyzer.split_dataset()

initial_instance = X[initial_instance_index]
initial_prediction = Y[initial_instance_index]
model = RandomForestClassifier()  # pluggable model that we train to explain.
np.delete(X, initial_instance_index)
np.delete(Y, initial_instance_index)
print("Training model to explain")
# model.fit(X, Y)
model.fit(X[:200, :], Y[:200])
print("Finished training")

counterfactuals, scores = bcg_xai.run(initial_instance, initial_prediction, target, data_analyzer, model)

predictions = model.predict(counterfactuals)
output = {
    "initial_instance": initial_instance.tolist(),
    "initial_prediction": str(initial_prediction),
    "target_type": target.target_type(),
    "target_value": target.target_value(),
    "target_feature": target.target_feature(),
    "total_time": str(time_measurement.total_time),
    "time_to_first_solution": str(time_measurement.time_to_first_solution),
    "time_to_best_solution": str(time_measurement.time_to_best_solution),
    "counterfactuals": counterfactuals.tolist(),
    "predictions": predictions.tolist()
}
output_filename = "algorithm_output.json"
with open(output_filename, 'w') as outfile:
    json.dump(output, outfile)

