# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD
import json
import time

import fatf.transparency.predictions.counterfactuals as fatf_cf
import fatf.utils.data.datasets as fatf_datasets
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common.DataAnalyzer import *
from common.Target import Target


# The dataset file must be formatted in the *comma separated value* (*csv*)
#     standard with ``,`` used as the delimiter. The first row of the file must
#     be a header formatted as follows:
#     ``n_samples,n_features,class_name_1,class_name_2,...``
def format_csv_fatf(file_name):
    dataset = pd.read_csv(csv_path)
    shapes = [dataset.shape[0], dataset.shape[1] - 1]
    target_column_name = dataset.columns[len(dataset.columns) - 1]
    target = dataset[target_column_name]
    categories = target.unique()
    data = dataset.values[:, :-1].astype(float)
    header = ','.join(str(e) for e in [*shapes, *categories])

    modified_csv = file_name + '.mod'
    with open(modified_csv, 'w') as write_obj:
        write_obj.write(header + '\n')

        i = 0
        while i < len(data):
            write_obj.write(', '.join(str(e) for e in data[i]) + ', ' + target[i] + '\n')
            i += 1
    # os.remove(file_name)


def explain(counterfactuals, predictions, target, X, Y):
    # print('\nCounterfactuals for the data point:')
    # pprint(dp_1_cfs)
    print("Generated counterfactuals {}".format(len(counterfactuals)))
    data_analyzer = DataAnalyzer(X, Y)
    similarity_calculator = SimilarityCalculator(initial_instance, initial_prediction, target, data_analyzer)
    print("Similarities")
    for counterfactual, prediction in zip(counterfactuals, predictions):
        score = similarity_calculator.calculate_scores(np.array([counterfactual]), np.array(prediction))
        print("Counterfactual with score {} (01) {}".format("%.3f" % score, counterfactual))


csv_path = "datasets/diabetes.csv"
target = Target("classification", "tested_negative")
format_csv_fatf(csv_path)

dataset = fatf_datasets.load_data(csv_path + ".mod")
X = np.array(dataset['data'])
Y = dataset['target']

clf = RandomForestClassifier()
# clf = fatf_models.KNN()
clf.fit(X, Y)

# Create a Counterfactual Explainer
t = time.process_time()
cf_explainer = fatf_cf.CounterfactualExplainer(
    model=clf,
    dataset=X,
    categorical_indices=[],
    default_numerical_step_size=0.1)

# Select a data point to be explained
initial_instance_index = 0
initial_instance = X[initial_instance_index]
initial_prediction = Y[initial_instance_index]

# Get a Counterfactual Explanation tuple for this data point
print("initial instance: {}, output: {}".format(initial_instance, initial_prediction))
explanation_tuple = cf_explainer.explain_instance(initial_instance)
total_time = time.process_time() - t
counterfactuals, distances, predictions = explanation_tuple

explain(counterfactuals, predictions, target, X, Y)

clf_predictions = clf.predict(counterfactuals)

output = {
    "initial_instance": initial_instance.tolist(),
    "initial_prediction": initial_prediction,
    "target_type": target.target_type(),
    "target_value": target.target_value(),
    "total_time": total_time,
    "counterfactuals": counterfactuals.tolist(),
    "predictions": clf_predictions.tolist()
}
output_filename = "fatf_output.json"
with open(output_filename, 'w') as outfile:
    json.dump(output, outfile)
