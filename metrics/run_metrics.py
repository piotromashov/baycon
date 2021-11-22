import pandas as pd

from InstancesMetrics import InstancesMetrics

# TODO: read from input path to dataset
dataset_filename = "datasets/diabetes.csv"
df = pd.read_csv(dataset_filename)

filename = "algorithm_output"
input_json_filename = filename + ".json"
output_csv_filename = filename + ".csv"

metrics = InstancesMetrics(df, input_json_filename)
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
