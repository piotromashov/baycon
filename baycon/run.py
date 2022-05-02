import json

import pandas as pd

import baycon.bayesian_generator as baycon
import baycon.time_measurement as time_measurement
from common.DataAnalyzer import *
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
        return class_prediction == self._target.target_value()


def prepare_model_and_data(dataset, model_name, target, categorical_features):
    dataframe = pd.read_csv("datasets/" + dataset + ".csv")
    Y = dataframe[[target.target_feature()]].values.ravel()
    X = dataframe.drop([target.target_feature()], axis=1).values

    if categorical_features:
        X = encode(X, categorical_features)
    if model_name == "RF":
        if target.target_type() == Target.TYPE_CLASSIFICATION:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
    else:
        from sklearn.model_selection import RandomizedSearchCV
        if target.target_type() == Target.TYPE_CLASSIFICATION:
            from sklearn.svm import SVC
            model = SVC()
        else:
            from sklearn.svm import SVR
            model = SVR()
        # normalize data
        X = scale(X)
        # tune parameters for SVM to increase precision
        Cs = [0.1, 1, 10, 100]
        gammas = [0.01, 0.1, 1]
        kernels = ['rbf', 'poly', 'sigmoid']
        param_grid = {'kernel': kernels, 'C': Cs, 'gamma': gammas}
        model = RandomizedSearchCV(model, param_grid, cv=5, n_jobs=-1)

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

    feature_names = dataframe.columns[dataframe.columns != target.target_feature()]
    return model, X, Y, feature_names


def execute(dataset_name, target, initial_instance_index, categorical_features=[], actionable_features=[]):
    total_runs = 1
    models_to_run = ["RF", "SVM"]
    for model_name in models_to_run:
        for run in range(total_runs):
            model, X, Y, feature_names = prepare_model_and_data(dataset_name, model_name, target, categorical_features)
            data_analyzer = DataAnalyzer(X, Y, feature_names, target, categorical_features, actionable_features)
            X, Y = data_analyzer.data()
            initial_instance = X[initial_instance_index]
            initial_prediction = Y[initial_instance_index]
            print("--- Executing: {} Initial Instance: {} Target: {} Model: {} Run: {} ---".format(
                dataset_name,
                initial_instance_index,
                target.target_value_as_string(),
                model_name,
                run
            ))
            counterfactuals, ranker = baycon.run(initial_instance, initial_prediction, target, data_analyzer, model)
            predictions = np.array([])
            try:
                predictions = model.predict(counterfactuals)
            except ValueError:
                pass
            output = {
                "initial_instance": initial_instance.tolist(),
                "initial_prediction": str(initial_prediction),
                "categorical_features": categorical_features,
                "actionable_features": actionable_features,
                "target_type": target.target_type(),
                "target_value": target.target_value(),
                "target_feature": target.target_feature(),
                "total_time": str(time_measurement.total_time),
                "time_to_first_solution": str(time_measurement.time_to_first_solution),
                "time_to_best_solution": str(time_measurement.time_to_best_solution),
                "counterfactuals": counterfactuals.tolist(),
                "predictions": predictions.tolist()
            }

            output_filename = "{}_{}_{}_{}_{}_{}.json".format("bcg", dataset_name, initial_instance_index,
                                                              target.target_value_as_string(), model_name, run)
            with open(output_filename, 'w') as outfile:
                json.dump(output, outfile)
            print("--- Finished: saved file {}\n".format(output_filename))


# --- BEGIN CLASSIFICATION EXPERIMENTS ---
t = Target(target_type="classification", target_feature="class", target_value="tested_positive")
execute("diabetes", t, 1)
execute("diabetes", t, 3)
execute("diabetes", t, 5)
execute("diabetes", t, 7)
execute("diabetes", t, 10)
t = Target(target_type="classification", target_feature="class", target_value="tested_negative")
execute("diabetes", t, 0)
execute("diabetes", t, 2)
execute("diabetes", t, 4)
execute("diabetes", t, 8)
execute("diabetes", t, 9)

t = Target(target_type="classification", target_feature="problems", target_value="yes")
execute("kc2", t, 9)
execute("kc2", t, 145)
execute("kc2", t, 252)
execute("kc2", t, 357)
execute("kc2", t, 413)
t = Target(target_type="classification", target_feature="problems", target_value="no")
execute("kc2", t, 4)
execute("kc2", t, 421)
execute("kc2", t, 485)
execute("kc2", t, 496)
execute("kc2", t, 520)

# df = pd.read_csv("datasets/phpGUrE90.csv")
# for column in df.columns:
#     print(column)
#     print(df[column].unique())
# cat_features = ["V3", "V4", "V5", "V6", "V7", "V9", "V10", "V11", "V16", "V19", "V20", "V21", "V23", "V24", "V25",
#               "V26", "V29", "V32", "V33", "V34", "V35", "V38", "V40", "V41"]
cat_features = []
t = Target(target_type="classification", target_feature="Class", target_value=2)
execute("phpGUrE90", t, 300, cat_features)
execute("phpGUrE90", t, 511, cat_features)
execute("phpGUrE90", t, 686, cat_features)
execute("phpGUrE90", t, 950, cat_features)
execute("phpGUrE90", t, 1024, cat_features)
t = Target(target_type="classification", target_feature="Class", target_value=1)
execute("phpGUrE90", t, 9, cat_features)
execute("phpGUrE90", t, 80, cat_features)
execute("phpGUrE90", t, 145, cat_features)
execute("phpGUrE90", t, 202, cat_features)
execute("phpGUrE90", t, 257, cat_features)

# cat_features = ["gender"]
# t = Target(target_type="classification", target_feature="class", target_value=1)
# for initial_instance_index in [33, 72, 99, 195, 309]:
#     execute("parkinsonspeech", t, initial_instance_index, cat_features)
# t = Target(target_type="classification", target_feature="class", target_value=0)
# for initial_instance_index in [0, 40, 110, 230, 480]:
#     execute("parkinsonspeech", t, initial_instance_index, cat_features)

# --- BEGIN REGRESSION EXPERIMENTS ---
cat_features = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
t = Target(target_type="regression", target_feature="cnt", target_value=[777, 1565])
execute("bike", t, 96, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(2184, 2972))
execute("bike", t, 156, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(3572, 4360))
execute("bike", t, 457, cat_features)

t = Target(target_type="regression", target_feature="cnt", target_value=(4717, 5505))
execute("bike", t, 96, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(6124, 6912))
execute("bike", t, 156, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(7512, 8300))
execute("bike", t, 457, cat_features)

t = Target(target_type="regression", target_feature="cnt", target_value=(float("-inf"), 1565))
execute("bike", t, 96, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(float("-inf"), 2972))
execute("bike", t, 156, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(float("-inf"), 4360))
execute("bike", t, 457, cat_features)

t = Target(target_type="regression", target_feature="cnt", target_value=(4717, float("inf")))
execute("bike", t, 96, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(6124, float("inf")))
execute("bike", t, 156, cat_features)
t = Target(target_type="regression", target_feature="cnt", target_value=(7512, float("inf")))
execute("bike", t, 457, cat_features)

cat_features = ["waterfront", "date_year"]
t = Target(target_type="regression", target_feature="price", target_value=(80487.5, 160975.0))
execute("housesales", t, 21065, cat_features)
t = Target(target_type="regression", target_feature="price", target_value=(208537.5, 289025.0))
execute("housesales", t, 48, cat_features)
t = Target(target_type="regression", target_feature="price", target_value=(403537.5, 484025.0))
execute("housesales", t, 915, cat_features)

t = Target(target_type="regression", target_feature="price", target_value=(float("-inf"), 160975.0))
execute("housesales", t, 21065, cat_features)
t = Target(target_type="regression", target_feature="price", target_value=(float("-inf"), 289025))
execute("housesales", t, 48, cat_features)
t = Target(target_type="regression", target_feature="price", target_value=(float("-inf"), 484025))
execute("housesales", t, 915, cat_features)

t = Target(target_type="regression", target_feature="price", target_value=(482925, float("inf")))
execute("housesales", t, 21065, cat_features)
t = Target(target_type="regression", target_feature="price", target_value=(610975, float("inf")))
execute("housesales", t, 48, cat_features)
t = Target(target_type="regression", target_feature="price", target_value=(805975, float("inf")))
execute("housesales", t, 915, cat_features)

# TODO: add in moc jsons the target_feature
t = Target(target_type="regression", target_feature="protein", target_value=(3.84375, 7.6625))
execute("tecator", t, 107)
t = Target(target_type="regression", target_feature="protein", target_value=(7.54375, 11.3625))
execute("tecator", t, 86)
t = Target(target_type="regression", target_feature="protein", target_value=(9.04375, 12.8625))
execute("tecator", t, 2)

t = Target(target_type="regression", target_feature="protein", target_value=(22.9375, 26.75625))
execute("tecator", t, 107)
t = Target(target_type="regression", target_feature="protein", target_value=(26.6375, 30.45625))
execute("tecator", t, 86)
t = Target(target_type="regression", target_feature="protein", target_value=(28.1375, 31.95625))
execute("tecator", t, 2)

t = Target(target_type="regression", target_feature="protein", target_value=(float("-inf"), 7.6625))
execute("tecator", t, 107)
t = Target(target_type="regression", target_feature="protein", target_value=(float("-inf"), 11.3625))
execute("tecator", t, 86)
t = Target(target_type="regression", target_feature="protein", target_value=(float("-inf"), 12.8625))
execute("tecator", t, 2)

t = Target(target_type="regression", target_feature="protein", target_value=(22.9375, float("inf")))
execute("tecator", t, 107)
t = Target(target_type="regression", target_feature="protein", target_value=(26.6375, float("inf")))
execute("tecator", t, 86)
t = Target(target_type="regression", target_feature="protein", target_value=(28.1375, float("inf")))
execute("tecator", t, 2)

# # EXTRA MNIST DATASET
# dataset = "mnist"
# t = Target(target_type="classification", target_feature="class", target_value=9)
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
