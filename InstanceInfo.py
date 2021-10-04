from collections import Counter
import numpy as np
import numpy_utils as npu

SCORE_JITTER = 0.75     # giving search space for the solutions we are finding


class InstancesInfo:
    def __init__(self, instances, model, initial_instance, dataconstraints, target):
        self._model = model
        self._initial_instance = initial_instance
        self._dataconstraints = dataconstraints
        self._target = target
        self._newBest = True
        self._instances = instances
        self._distances = []
        self._scores = []
        self.calculate_objective_all()

    def calculate_objective_all(self):
        # obtain model prediction on those values
        Y = np.array(self._model.predict(self._instances))
        max_distance = self._dataconstraints.features_max_distance()
        # here should go the cost of attribute changes and their weights
        distances = npu.distance_arr(self._instances, self._initial_instance)
        # closeness to feature space of the potential counterfactual to the initial instance.
        similarity = 1 - distances / max_distance
        # check if we are moving towards the target or not.
        # if we are not moving towards the target, this is weighted as 0
        targets_achieved = Y == self._target
        scores = similarity * targets_achieved
        self._distances = distances
        self._scores = scores

    def __len__(self):
        return len(self._instances)

    def best(self):
        index = np.argmax(self._scores)
        return self._instances[index], self._distances[index], self._scores[index]

    def has_new_best(self):
        return self._newBest

    def achieved_target_count(self):
        return np.count_nonzero(self._scores > 0)

    def extend(self, instances_info):
        instances, distances, scores = instances_info.info()
        self._newBest = np.max(scores) > np.max(self._scores)
        self._instances = np.concatenate((self._instances, instances))
        self._distances = np.concatenate((self._distances, distances), axis=None)
        self._scores = np.concatenate((self._scores, scores), axis=None)

    def achieved_target_summary(self):
        achieved_indexes = self._scores > 0
        achieved_distances = self._distances[achieved_indexes]
        achieved_scores = self._scores[achieved_indexes]

        distances_counter = Counter(achieved_distances)
        sorted_distances = sorted(distances_counter.items(), key=lambda pair: pair[0])
        representation = []
        for k, (distance, count) in enumerate(sorted_distances):
            # find index of distance in the original array, and use that index to get the score
            index = achieved_distances.tolist().index(distance)
            score = achieved_scores[index]
            representation.append((distance, count, score))
        return representation

    def near(self, instance_score):
        near_best_index = self._scores > instance_score * SCORE_JITTER
        return self._instances[near_best_index]

    def info(self):
        return self._instances, self._distances, self._scores

    def instances(self):
        return self._instances
