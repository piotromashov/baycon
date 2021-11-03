import json

import pandas as pd

from DataAnalyzer import *
from InstancesMetrics import InstancesMetrics

# TODO: read from input path to dataset
dataset_filename = "datasets/kc2.csv"
dataset = pd.read_csv(dataset_filename)
data = np.array(dataset.values[:, :-1])
data_analyzer = DataAnalyzer(data)
distance_calculator = data_analyzer.distance_calculator().gower

algorithm_output_filename = "algorithm_output.json"

with open(algorithm_output_filename) as json_file:
    data = json.load(json_file)
    initial_instance = np.array(data["initial_instance"])
    counterfactuals = np.array(data["counterfactuals"])
    time_to_first_solution = data["time_to_first_solution"]
    elapsed_time = data["total_time"]

metrics = InstancesMetrics(initial_instance, counterfactuals, distance_calculator, time_to_first_solution, elapsed_time)
print(metrics)
