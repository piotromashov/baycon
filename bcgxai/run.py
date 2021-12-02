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


def prepare_model(dataset, model_name, X, Y, target_type):
    if model_name == "RF":
        if target_type == Target.TYPE_CLASSIFICATION:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
    else:
        if target_type == Target.TYPE_CLASSIFICATION:
            from sklearn.svm import SVC
            model = SVC()
        else:
            from sklearn.svm import SVR
            model = SVR()

    import pickle
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


def execute(dataset, target, initial_instance_index, cat_features=[]):
    # load dataset, train model
    df = pd.read_csv("datasets/" + dataset + ".csv")
    data_analyzer = DataAnalyzer(df, target, cat_features)
    data_analyzer.encode()
    X, Y = data_analyzer.data()

    initial_instance = X[initial_instance_index]
    initial_prediction = Y[initial_instance_index]
    run = "0"
    models_to_run = ["RF", "SVM"]
    for model_name in models_to_run:
        print("--- Executing: {} Initial Instance: {} Target: {} Model: {}---".format(
            dataset,
            initial_instance_index,
            target.target_value(),
            model_name
        ))
        model = prepare_model(dataset, model_name, X, Y, target.target_type())
        counterfactuals, scores = bcg_xai.run(initial_instance, initial_prediction, target, data_analyzer, model)
        # counterfactuals = data_analyzer.decode(counterfactuals)

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

        target_value = target.target_value()
        target_value = "{}-{}".format(target_value[0], target_value[1]) if isinstance(target_value, tuple) else str(
            target_value)
        output_filename = "{}_{}_{}_{}_{}_{}.json".format("bcg", dataset, initial_instance_index, model_name,
                                                          target_value,
                                                          run)
        with open(output_filename, 'w') as outfile:
            json.dump(output, outfile)
        print("--- Finished: saved file {}".format(output_filename))


# --- BEGIN CLASSIFICATION EXPERIMENTS ---
target = Target(target_type="classification", target_feature="class", target_value="tested_positive")
execute("diabetes", target, initial_instance_index=0)
execute("diabetes", target, initial_instance_index=2)
execute("diabetes", target, initial_instance_index=4)
execute("diabetes", target, initial_instance_index=6)
target = Target(target_type="classification", target_feature="class", target_value="tested_negative")
execute("diabetes", target, initial_instance_index=1)
execute("diabetes", target, initial_instance_index=3)
execute("diabetes", target, initial_instance_index=5)
execute("diabetes", target, initial_instance_index=7)
execute("diabetes", target, initial_instance_index=9)

target = Target(target_type="classification", target_feature="problems", target_value="yes")
for initial_instance_index in [8, 144, 251, 356, 412]:
    execute("kc2", target, initial_instance_index)
target = Target(target_type="classification", target_feature="problems", target_value="no")
for initial_instance_index in [3, 420, 484, 495, 519]:
    execute("kc2", target, initial_instance_index)

# TODO: agregarle los features categoricos
target = Target(target_type="classification", target_feature="Class", target_value=2)
for initial_instance_index in [299, 510, 685, 949, 1023]:
    execute("phpGUrE90", target, initial_instance_index)
target = Target(target_type="classification", target_feature="Class", target_value=1)
for initial_instance_index in [8, 79, 144, 201, 256]:
    execute("phpGUrE90", target, initial_instance_index)

target = Target(target_type="classification", target_feature="class", target_value=1)
for initial_instance_index in [33, 72, 99, 195, 309]:
    execute("pd_speech_features", target, initial_instance_index)
target = Target(target_type="classification", target_feature="class", target_value=0)
for initial_instance_index in [0, 40, 110, 230, 480]:
    execute("pd_speech_features", target, initial_instance_index)

# --- BEGIN REGRESSION EXPERIMENTS ---
# TODO: what are those infinity ranges? error throwing: invalud value encountered total_diff = np.abs(max_value - min_value)
cat_features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
target = Target(target_type="regression", target_feature="cnt", target_value=(777, 1565))
execute("bike", target, 95, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(2184, 2972))
execute("bike", target, 155, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(3572, 4360))
execute("bike", target, 456, cat_features)

target = Target(target_type="regression", target_feature="cnt", target_value=(4717, 5505))
execute("bike", target, 95, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(6124, 6912))
execute("bike", target, 155, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(7512, 8300))
execute("bike", target, 456, cat_features)

target = Target(target_type="regression", target_feature="cnt", target_value=(float("-inf"), 1565))
execute("bike", target, 95, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(float("-inf"), 2972))
execute("bike", target, 155, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(float("-inf"), 4360))
execute("bike", target, 456, cat_features)

target = Target(target_type="regression", target_feature="cnt", target_value=(4717, float("inf")))
execute("bike", target, 95, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(6124, float("inf")))
execute("bike", target, 155, cat_features)
target = Target(target_type="regression", target_feature="cnt", target_value=(7512, float("inf")))
execute("bike", target, 456, cat_features)

cat_features = ["waterfront", "date_year"]
target = Target(target_type="regression", target_feature="price", target_value=(80487.5, 160975.0))
execute("housesales", target, 21064, cat_features)
target = Target(target_type="regression", target_feature="price", target_value=(208537.5, 289025.0))
execute("housesales", target, 47, cat_features)
target = Target(target_type="regression", target_feature="price", target_value=(403537.5, 484025.0))
execute("housesales", target, 914, cat_features)

target = Target(target_type="regression", target_feature="price", target_value=(float("-inf"), 160975.0))
execute("housesales", target, 21064, cat_features)
target = Target(target_type="regression", target_feature="price", target_value=(float("-inf"), 289025))
execute("housesales", target, 47, cat_features)
target = Target(target_type="regression", target_feature="price", target_value=(float("-inf"), 484025))
execute("housesales", target, 914, cat_features)

target = Target(target_type="regression", target_feature="price", target_value=(482925, float("inf")))
execute("housesales", target, 21064, cat_features)
target = Target(target_type="regression", target_feature="price", target_value=(610975, float("inf")))
execute("housesales", target, 47, cat_features)
target = Target(target_type="regression", target_feature="price", target_value=(805975, float("inf")))
execute("housesales", target, 914, cat_features)

# TODO: add in moc jsons the target_feature
target = Target(target_type="regression", target_feature="protein", target_value=(3.84375, 7.6625))
execute("tecator", target, 106)
target = Target(target_type="regression", target_feature="protein", target_value=(7.54375, 11.3625))
execute("tecator", target, 86)
target = Target(target_type="regression", target_feature="protein", target_value=(9.04375, 12.8625))
execute("tecator", target, 2)

target = Target(target_type="regression", target_feature="protein", target_value=(22.9375, 26.75625))
execute("tecator", target, 106)
target = Target(target_type="regression", target_feature="protein", target_value=(26.6375, 30.45625))
execute("tecator", target, 86)
target = Target(target_type="regression", target_feature="protein", target_value=(28.1375, 31.95625))
execute("tecator", target, 2)

target = Target(target_type="regression", target_feature="protein", target_value=(float("-inf"), 7.6625))
execute("tecator", target, 106)
target = Target(target_type="regression", target_feature="protein", target_value=(float("-inf"), 11.3625))
execute("tecator", target, 86)
target = Target(target_type="regression", target_feature="protein", target_value=(float("-inf"), 12.8625))
execute("tecator", target, 2)

target = Target(target_type="regression", target_feature="protein", target_value=(22.9375, float("inf")))
execute("tecator", target, 106)
target = Target(target_type="regression", target_feature="protein", target_value=(26.6375, float("inf")))
execute("tecator", target, 86)
target = Target(target_type="regression", target_feature="protein", target_value=(28.1375, float("inf")))
execute("tecator", target, 2)

## EXTRA MNIST DATASET
# dataset = "mnist"
# target = Target(target_type="classification", target_feature="class", target_value=9)
# initial_instance_index = 0
# cat_features = []
#
# define black-box model
# from keras.layers import Dense, Input
# from keras.models import Model
# num_classes = 10
#
# _input = Input(shape=(X.shape[1],))
# x = Dense(512)(_input)
# x = Dense(512)(x)
# output = Dense(num_classes, activation='softmax')(x)
# model = Model(inputs=_input, outputs=output)
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X, Y, epochs=5, batch_size=2000)  # train the black-box model
# return ModelWrapper(model, target)
