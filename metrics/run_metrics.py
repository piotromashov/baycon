import pandas as pd

from InstancesMetrics import InstancesMetrics


def execute(input_json, dataset_path, cat_features):
    print("--- Running metrics on {} ---".format(input_json))
    df = pd.read_csv(dataset_path)
    InstancesMetrics(df, input_json, cat_features)

import glob

dataset_folder = "datasets/"
for experiment_file in glob.iglob('*.json'):
    dataset_name = experiment_file.split("_")[1]
    dataset_filename = dataset_name + ".csv"
    cat_features = None
    # TODO: move categorical_features values into json output of bcg run and read in InstanceMetrics
    if dataset_name == "bike":
        cat_features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
    elif dataset_name == "housesales":
        cat_features = ["waterfront", "date_year"]
    execute(experiment_file, dataset_folder + dataset_filename, cat_features)
