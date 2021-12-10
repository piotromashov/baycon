import glob

import pandas as pd

from InstancesMetrics import InstancesMetrics

dataset_folder = "datasets/"
for experiment_file in glob.iglob('*.json'):
    dataset_name = experiment_file.split("_")[1]
    model = experiment_file.split("_")[-2]
    dataset_filename = dataset_name + ".csv"

    print("--- Running metrics on {} ---".format(experiment_file))
    df = pd.read_csv(dataset_folder + dataset_filename)
    InstancesMetrics(df, experiment_file, model)
