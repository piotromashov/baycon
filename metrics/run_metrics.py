import pandas as pd

from InstancesMetrics import InstancesMetrics


def execute(input_json, dataset_path):
    print("--- Executing {} ---".format(input_json))
    df = pd.read_csv(dataset_path)
    metrics = InstancesMetrics(df, input_json)
    print(metrics)
    metrics.to_csv(input_json[:-4] + ".csv")
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.set_theme(style="whitegrid")
    # counterfactuals = pd.read_csv(output_csv_filename)
    # ax = sns.boxplot(counterfactuals["distance_x"])
    # plt.show()
    # ax = sns.boxplot(counterfactuals["features_changed"])
    # plt.show()


dataset_folder = "datasets/"

# CLASSIFICATION
dataset_filename = "diabetes.csv"
experiment_name = "bcg_diabetes_0_RF_tested_negative_0.json"
execute(experiment_name, dataset_folder + dataset_filename)

dataset_filename = "kc2.csv"
experiment_name = "bcg_kc2_4_RF_no_0.json"
execute(experiment_name, dataset_folder + dataset_filename)

dataset_filename = "pd_speech_features.csv"
experiment_name = "bcg_pd_speech_features_0_RF_0_0.json"
execute(experiment_name, dataset_folder + dataset_filename)

# REGRESSION
dataset_filename = "bike.csv"
experiment_name = "bcg_bike_0_RF_1500-2000_0.json"
execute(experiment_name, dataset_folder + dataset_filename)

dataset_filename = "house_sales.csv"
experiment_name = "bcg_house_sales_0_RF_200000-300000_0.json"
execute(experiment_name, dataset_folder + dataset_filename)
