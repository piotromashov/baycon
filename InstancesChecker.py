import numpy as np
import numpy_utils as npu
import acquisition_functions as acq_functions

# TODO: update this class to InstancesRanker, remove all unused initialization variables
class InstancesChecker:
    def __init__(self, objective_model, surrogate_model, initial_instance, dataconstraints, target):
        self._objective_model = objective_model
        self._surrogate_model = surrogate_model
        self._initial_instance = initial_instance
        self._dataconstraints = dataconstraints
        self._target = target

    def train_surrogate(self, instances, scores):
        self._surrogate_model.fit(instances, scores)

    def surrogate(self):
        return self._surrogate_model

    def rank(self, known_instances, instances_to_check, top_ranked):
        return self.opt_acquisition(known_instances, instances_to_check, top_ranked)

    def rank_with_objective(self, known_instances, instances_to_check, top_ranked):
        Y = np.array(self._objective_model.predict(instances_to_check))
        max_distance = self._dataconstraints.features_max_distance()
        # here should go the cost of attribute changes and their weights
        distances = npu.distance_arr(instances_to_check, self._initial_instance)
        # closeness to feature space of the potential counterfactual to the initial instance.
        similarity = 1 - distances / max_distance
        # check if we are moving towards the target or not.
        # if we are not moving towards the target, this is weighted as 0
        targets_achieved = Y == self._target
        score = similarity * targets_achieved
        sorted_index = np.argsort(score)
        return instances_to_check[sorted_index][-top_ranked:]

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

    # returns scores caclulated with an acquisition function (see acqusition_functions.py)
    def acquisition(self, known_instances, instances_to_check):
        mu, _ = self.get_ensemble_scores(known_instances)
        best_mu = max(mu)
        mu, std = self.get_ensemble_scores(instances_to_check)
        score = acq_functions.EI(mu, std, best_mu, epsilon=.001)
        return score

    # select top_ranked alternatives based on the acquisition function
    def opt_acquisition(self, known_instances, instances_to_check, top_ranked):
        # calculate the acquisition function for each candidate
        scores = self.acquisition(known_instances, instances_to_check)
        # locate the index of the largest scores
        top_ranked = len(scores) if top_ranked > len(scores) else top_ranked
        best_alternatives_index = np.argpartition(scores, -top_ranked)[-top_ranked:]  # get top_n candidates
        return instances_to_check[best_alternatives_index].copy()
