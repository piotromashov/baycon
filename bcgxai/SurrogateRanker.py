import numpy as np

import bcgxai.acquisition_functions as acq_functions

TOP_RANKED = 20
OVERSAMPLING_AMOUNT = 10
AGGREGATED_FEATURES = 2


class SurrogateRanker:
    def __init__(self, objective_model, surrogate_model, initial_X, score_calculator, target):
        self._objective_model = objective_model
        self._surrogate_model = surrogate_model
        self._initial_X = initial_X
        # self._score_calculator = score_calculator
        self._target = target
        # two features more for feature changes and total_difference
        self._X = np.empty(shape=(0, self._initial_X.shape[0] + AGGREGATED_FEATURES))
        self._Y = np.array([])

    def train(self):
        print("Re-training surrogate model with data size: {}".format(self._X.shape[0]))
        self._surrogate_model.fit(self._X, self._Y)

    def surrogate(self):
        return self._surrogate_model

    def rank(self, known_instances, instances_to_check):
        known_X = self.prepare_x(known_instances)
        X_to_check = self.prepare_x(instances_to_check)
        _, index = self.opt_acquisition(known_X, X_to_check)
        return instances_to_check[index]

    # def rank_with_objective(self, known_instances, instances_to_check):
    #     predictions = np.array(self._objective_model.predict(instances_to_check))
    #     scores = self._score_calculator.calculate_score(instances_to_check, predictions)
    #     sorted_index = np.argsort(scores)
    #     return instances_to_check[sorted_index][-TOP_RANKED:]

    # returns mean values and standard deviation calculated over the predictions
    # from each separate model from a given ensemble models
    def get_ensemble_scores(self, instances):
        ens_predictions = []
        for est in range(len(self._surrogate_model.estimators_)):
            ens_predictions.append(self._surrogate_model.estimators_[est].predict(instances))
        ens_predictions = np.array(ens_predictions)

        mu = ens_predictions.mean(axis=0)
        std = ens_predictions.std(axis=0)
        return mu, std

    # returns scores calculated with an acquisition function (see acquisition_functions.py)
    def acquisition(self, known_instances, instances_to_check):
        mu, _ = self.get_ensemble_scores(known_instances)
        best_mu = max(mu)
        mu, std = self.get_ensemble_scores(instances_to_check)
        score = acq_functions.EI(mu, std, best_mu, epsilon=.001)
        return score

    # select top_ranked alternatives based on the acquisition function
    def opt_acquisition(self, known_instances, instances_to_check):
        # calculate the acquisition function for each candidate
        scores = self.acquisition(known_instances, instances_to_check)
        # locate the index of the largest scores
        top_ranked = len(scores) if TOP_RANKED > len(scores) else TOP_RANKED
        best_alternatives_index = np.argpartition(scores, -top_ranked)[-top_ranked:]  # get top_n candidates
        return instances_to_check[best_alternatives_index].copy(), best_alternatives_index

    def prepare_x(self, X):
        differences = X - self._initial_X
        differences_sum = np.sum(np.abs(differences), axis=1)
        features_equal_to_initial = np.sum(X == self._initial_X, axis=1)
        similarity_and_differences = np.array([differences_sum, features_equal_to_initial]).transpose()
        surrogate_X = np.concatenate((differences, similarity_and_differences), axis=1)
        return surrogate_X

    def update(self, X, Y):
        self._X = np.concatenate((self._X, self.prepare_x(X)))
        self._Y = np.concatenate((self._Y, Y), axis=None)

    def oversample_update(self, X, Y):
        oversampled_X = np.repeat([X], OVERSAMPLING_AMOUNT, axis=0)
        oversampled_Y = np.repeat(Y, OVERSAMPLING_AMOUNT)
        self.update(oversampled_X, oversampled_Y)
