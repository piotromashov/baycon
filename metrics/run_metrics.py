import pandas as pd

from InstancesMetrics import InstancesMetrics


def execute(input_json, dataset_path):
    print("--- Running metrics on {} ---".format(input_json))
    df = pd.read_csv(dataset_path)
    metrics = InstancesMetrics(df, input_json)
    print(metrics)
    metrics.to_csv(input_json[:-5] + ".csv")
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.set_theme(style="whitegrid")
    # counterfactuals = pd.read_csv(output_csv_filename)
    # ax = sns.boxplot(counterfactuals["distance_x"])
    # plt.show()
    # ax = sns.boxplot(counterfactuals["features_changed"])
    # plt.show()


import glob

dataset_folder = "datasets/"
for experiment_file in glob.iglob('*.json'):
    dataset_filename = experiment_file.split("_")[1] + ".csv"
    execute(experiment_file, dataset_folder + dataset_filename)

# dataset_filename = "housesales.csv"
# experiment_name = "bcg_house_sales_0_RF_200000-300000_0.json"
# execute(experiment_name, dataset_folder + dataset_filename)
