import numpy as np
import numpy_utils as npu
import acquisition_functions as acq_functions

TOP_RANKED = 20
OVERSAMPLING_AMOUNT = 10
TRAINING_GROWTH = 5         # should not be a % of growth, because we are using a fixed top 20 ranked for now.


class SurrogateRanker:
    def __init__(self, objective_model, surrogate_model, initial_instance, distance_calculator, target):
        self._objective_model = objective_model
        self._surrogate_model = surrogate_model
        self._initial_instance = initial_instance
        self._distance_calculator = distance_calculator
        self._target = target
        self._X = np.array([], dtype=np.int64).reshape(0, self._initial_instance.shape[0])
        self._Y = np.array([])
        self._updated_train_achieved_target = False

    def train(self):
        if not self._updated_train_achieved_target:
            return
        print("Re-training surrogate model with data size: {}".format(self._X.shape[0]))
        self._surrogate_model.fit(self._X, self._Y)

    def surrogate(self):
        return self._surrogate_model

    def rank(self, known_instances, instances_to_check):
        return self.opt_acquisition(known_instances, instances_to_check)

    def rank_with_objective(self, known_instances, instances_to_check):
        Y = np.array(self._objective_model.predict(instances_to_check))
        # closeness to feature space of the potential counterfactual to the initial instance.
        similarity = 1 - self._distance_calculator.gower(self._initial_instance, instances_to_check)
        # if we are not moving towards the target, this is weighted as 0
        targets_achieved = Y == self._target
        score = similarity * targets_achieved
        sorted_index = np.argsort(score)
        return instances_to_check[sorted_index][-TOP_RANKED:]

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
        return instances_to_check[best_alternatives_index].copy()

    def update(self, instances, scores):
        self._X = np.concatenate((self._X, instances))
        self._Y = np.concatenate((self._Y, scores), axis=None)
        self._updated_train_achieved_target = len(scores[scores > 0]) > 0

    def oversample_update(self, instance, score):
        oversampled_instance = np.repeat([instance], OVERSAMPLING_AMOUNT, axis=0)
        oversampled_score = np.repeat(score, OVERSAMPLING_AMOUNT)
        self.update(oversampled_instance, oversampled_score)
