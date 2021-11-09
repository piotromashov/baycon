import pandas as pd

from InstancesMetrics import InstancesMetrics
from common.DataAnalyzer import *

# TODO: read from input path to dataset
dataset_filename = "datasets/diabetes.csv"
dataset = pd.read_csv(dataset_filename)
data = np.array(dataset.values[:, :-1])
data_analyzer = DataAnalyzer(data)
distance_calculator = data_analyzer.distance_calculator().gower

filename = "algorithm_output"
input_json_filename = filename + ".json"
output_csv_filename = filename + ".csv"

metrics = InstancesMetrics(input_json_filename, distance_calculator)
print(metrics)
metrics.to_csv(output_csv_filename)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
counterfactuals = pd.read_csv(output_csv_filename)
ax = sns.boxplot(counterfactuals["distance_x"])
plt.show()
ax = sns.boxplot(counterfactuals["features_changed"])
plt.show()
