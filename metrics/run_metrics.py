import pandas as pd

from InstancesMetrics import InstancesMetrics
from common.DataAnalyzer import *

# TODO: read from input path to dataset
dataset_filename = "datasets/diabetes.csv"
df = pd.read_csv(dataset_filename)
X = np.array(df.values[:, :-1])
Y = np.array(df.values[:, -1])
data_analyzer = DataAnalyzer(X, Y)

filename = "algorithm_output"
input_json_filename = filename + ".json"
output_csv_filename = filename + ".csv"

metrics = InstancesMetrics(input_json_filename, data_analyzer)
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
