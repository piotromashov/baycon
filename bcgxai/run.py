import json

import numpy as np
import pandas as pd

import bcgxai.bayesian_generator as bcg_xai
import bcgxai.time_measurement as time_measurement
from common.DataAnalyzer import DataAnalyzer
from common.Target import Target


class ModelWrapper:
    def __init__(self, model, target):
        self._target = target
        self._model = model

    def predict(self, instances):
        try:
            predictions = self._model.predict(instances)
        except ValueError:
            return np.array([0])
        class_prediction = np.argmax(predictions, axis=1)
        return class_prediction == target.target_value()


def prepare_model(dataset, X, Y):
    import pickle
    if dataset != 'mnist':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        # model = KNeighborsClassifier()
        model_filename = "{0}_{1}.sav".format(type(model).__name__, dataset)
        try:
            print("Checking if {} exists, loading...".format(model_filename))
            model = pickle.load(open(model_filename, 'rb'))
            print("Loaded model")
        except FileNotFoundError:
            print("Not found, Training model to explain")
            model.fit(X, Y)
            print("Finished training")
            pickle.dump(model, open(model_filename, 'wb'))
        return model

    # define black-box model
    from keras.layers import Dense, Input
    from keras.models import Model
    num_classes = 10

    _input = Input(shape=(X.shape[1],))
    x = Dense(512)(_input)
    x = Dense(512)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=_input, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=5, batch_size=2000)  # train the black-box model
    return ModelWrapper(model, target)


# dataset = "diabetes"
# target = Target(target_type="classification", target_feature="class", target_value="tested_negative")
# initial_instance_index = 0
# cat_features = []

# dataset = "kc2"
# target = Target(target_type="classification", target_feature="problems", target_value="no")
# initial_instance_index = 4
# cat_features = []

# dataset = "pd_speech_features"
# target = Target(target_type="classification", target_feature="class", target_value=0)
# initial_instance_index = 0
# cat_features = []

# dataset = "house_sales"
# target = Target(target_type="regression", target_feature="price", target_value=(200000, 300000))
# initial_instance_index = 0
# cat_features = ["waterfront", "date_year"]

dataset = "bike"
target = Target(target_type="regression", target_feature="cnt", target_value=(1500, 2000))
initial_instance_index = 0
cat_features = []

# dataset = "mnist"
# target = Target(target_type="classification", target_feature="class", target_value=9)
# initial_instance_index = 0
# cat_features = []

# load dataset, train model
df = pd.read_csv("datasets/" + dataset + ".csv")
data_analyzer = DataAnalyzer(df, target=target, cat_features=cat_features)
data_analyzer.encode()
X, Y = data_analyzer.data()

initial_instance = X[initial_instance_index]
initial_prediction = Y[initial_instance_index]
model = prepare_model(dataset, X[:10000], Y[:10000])

counterfactuals, scores = bcg_xai.run(initial_instance, initial_prediction, target, data_analyzer, model)

predictions = np.array([])
try:
    predictions = model.predict(counterfactuals)
except ValueError:
    pass
output = {
    "initial_instance": initial_instance.tolist(),
    "initial_prediction": str(initial_prediction),
    "target_type": target.target_type(),
    "target_value": target.target_value(),
    "target_feature": target.target_feature(),
    "total_time": str(time_measurement.total_time),
    "time_to_first_solution": str(time_measurement.time_to_first_solution),
    "time_to_best_solution": str(time_measurement.time_to_best_solution),
    "counterfactuals": counterfactuals.tolist(),
    "predictions": predictions.tolist()
}
output_filename = "algorithm_output.json"
with open(output_filename, 'w') as outfile:
    json.dump(output, outfile)
