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


def custom_model(dataset_filename, X, Y):
    if dataset_filename != 'datasets/mnist.csv':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        # model = KNeighborsClassifier()
        print("Training model to explain")
        model.fit(X, Y)
        print("Finished training")
        return model

    # define black-box model
    from keras.layers import Dense, Input
    from keras.models import Model
    num_classes = 10

    # transform to tabular
    # x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    _input = Input(shape=(X.shape[1],))
    x = Dense(512)(_input)
    x = Dense(512)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=_input, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train the black-box model
    # model.fit(X, Y, epochs=5, batch_size=2000)
    model.fit(X, Y, epochs=5, batch_size=2000)
    return ModelWrapper(model, target)


# dataset_filename = "datasets/diabetes.csv"
# target = Target(target_type="classification", target_feature="class", target_value="tested_negative")
# initial_instance_index = 0

# dataset_filename = "datasets/kc2.csv"
# target = Target(target_type="classification", target_feature="problems", target_value="no")
# initial_instance_index = 4

# dataset_filename = "datasets/pd_speech_features.csv"
# target = Target(target_type="classification", target_feature="class", target_value=0)
# initial_instance_index = 0

dataset_filename = "datasets/house_sales.csv"
target = Target(target_type="regression", target_feature="price", target_value="increase")
initial_instance_index = 0
cat_features = ["waterfront", "date_year"]

# dataset_filename = "datasets/mnist.csv"
# target = Target(target_type="classification", target_feature="class", target_value=9)
# initial_instance_index = 0

# load dataset, train model
df = pd.read_csv(dataset_filename)
data_analyzer = DataAnalyzer(df, target=target)
X, Y = data_analyzer.split_dataset()

initial_instance = X[initial_instance_index]
initial_prediction = Y[initial_instance_index]
# np.delete(Y, initial_instance_index)
# np.delete(X, initial_instance_index)
model = custom_model(dataset_filename, X, Y)

counterfactuals, scores = bcg_xai.run(initial_instance, initial_prediction, target, data_analyzer, model)

predictions = model.predict(counterfactuals)
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
