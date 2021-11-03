"""
=========================================
Using Counterfactual Prediction Explainer
=========================================

This example illustrates how to use the Counterfactual Prediction explainer
(:class:`fatf.transparency.predictions.counterfactuals.\
CounterfactualExplainer`) and how to interpret the 3-tuple that it returns by
"textualising" it (:func:`fatf.transparency.predictions.counterfactuals.\
textualise_counterfactuals`).
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import fatf.transparency.predictions.counterfactuals as fatf_cf
import fatf.utils.data.datasets as fatf_datasets
import fatf.utils.models as fatf_models
import numpy as np
import pandas as pd

print(__doc__)


# The dataset file must be formatted in the *comma separated value* (*csv*)
#     standard with ``,`` used as the delimiter. The first row of the file must
#     be a header formatted as follows:
#     ``n_samples,n_features,class_name_1,class_name_2,...``
def format_csv_fatf(file_name):
    dataset = pd.read_csv(csv_path)
    shapes = [dataset.shape[0] - 1, dataset.shape[1] - 1]
    target_column_name = dataset.columns[len(dataset.columns) - 1]
    target = dataset[target_column_name]
    categories = target.unique()
    data = dataset.values[:, :-1].astype(float)
    header = ','.join(str(e) for e in [*shapes, *categories])

    modified_csv = file_name + '.mod'
    with open(modified_csv, 'w') as write_obj:
        write_obj.write(header + '\n')

        i = 1  # start after the header
        while i < len(data):
            write_obj.write(', '.join(str(e) for e in data[i]) + ', ' + target[i] + '\n')
            i += 1
    # os.remove(file_name)


csv_path = "../datasets/kc2.csv"
format_csv_fatf(csv_path)

dataset = fatf_datasets.load_data(csv_path + ".mod")
X = np.array(dataset['data'])
Y = dataset['target']

# dataset = fatf_datasets.load_iris()
# X = np.array(dataset['data'])
# Y = dataset['target']

# dataset = fetch_openml(name='kc2', version=1)
# X = np.array(dataset['data'])
# Y = dataset['target']

# Train a model

# clf = RandomForestClassifier()
clf = fatf_models.KNN()
clf.fit(X, Y)

# Create a Counterfactual Explainer
cf_explainer = fatf_cf.CounterfactualExplainer(
    model=clf,
    dataset=X,
    categorical_indices=[],
    default_numerical_step_size=0.1)

# Select a data point to be explained
dp_1_index = 3
dp_1_X = X[dp_1_index, :]
dp_1_y = Y[dp_1_index]

# Get a Counterfactual Explanation tuple for this data point
print("initial instance: {}, output: {}".format(dp_1_X, dp_1_y))
dp_1_cf_tuple = cf_explainer.explain_instance(dp_1_X)
dp_1_cfs, dp_1_cfs_distances, dp_1_cfs_predictions = dp_1_cf_tuple

print("Generated counterfactuals {}".format(len(dp_1_cfs)))
# print('\nCounterfactuals for the data point:')
# pprint(dp_1_cfs)

from DataAnalyzer import *

data_analyzer = DataAnalyzer(X)
distance_calculator = data_analyzer.distance_calculator()
print("1 - Gower distances")
for key, counterfactual in enumerate(dp_1_cfs):
    score = 1 - distance_calculator.gower(dp_1_X, np.array([counterfactual]))
    print("Counterfactual with score {} (01) {}".format("%.4f" % score, counterfactual))
