import pandas as pd

from DataAnalyzer import *
from InstancesMetrics import InstancesMetrics

# TODO: read from input path to dataset
dataset_filename = "datasets/kc2.csv"
dataset = pd.read_csv(dataset_filename)
data = np.array(dataset.values[:, :-1])
data_analyzer = DataAnalyzer(data)
distance_calculator = data_analyzer.distance_calculator().gower

# TODO: read from file initial instance and all counterfactuals, calculate metrics and present them
instances_filename = "instances.csv"
instances = pd.read_csv(instances_filename).values[:, :-1]

initial_instance = instances[0]
counterfactuals = instances[1:]

metrics = InstancesMetrics(initial_instance, counterfactuals, distance_calculator)
print(metrics)
