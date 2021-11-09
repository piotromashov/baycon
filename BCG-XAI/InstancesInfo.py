import time
from collections import Counter

import numpy as np

import time_measurement

SCORE_JITTER = 0.75  # giving search space for the solutions we are finding
MINIMUM_SCORE = 0


class InstancesInfo:
    # TODO: remove score and use distance
    # TODO: save if target have been achieved, move scores to a method instead and calculate
    def __init__(self, instances, model, initial_instance, distance_calculator, target):
        self._model = model
        self._initial_instance = initial_instance
        self._target = target
        self._newBest = True
        self._instances = instances
        self._similarity_x = []
        self._scores = []
        self._distance_calculator = distance_calculator
        self.calculate_objective_all()

    def calculate_objective_all(self):
        # obtain model prediction on those values
        Y = np.array(self._model.predict(self._instances))
        # closeness to feature space of the potential counterfactual to the initial instance.
        self._similarity_x = 1 - self._distance_calculator.gower(self._initial_instance, self._instances)
        # if we are not moving towards the target, this is weighted as 0
        similarity_y = Y == self._target
        scores = self._similarity_x * similarity_y
        self._scores = scores

        if scores[scores > 0].any() and not time_measurement.first_solution_clock:
            time_measurement.first_solution_clock = time.process_time()

    def __len__(self):
        return len(self._instances)

    def best(self):
        index = np.argmax(self._scores)
        return self._instances[index], self._similarity_x[index], self._scores[index]

    def has_new_best(self):
        return self._newBest

    def achieved_target_count(self):
        return np.count_nonzero(self._scores > MINIMUM_SCORE)

    def extend(self, instances_info):
        instances, distances, scores = instances_info.info()
        self._newBest = np.max(scores) > np.max(self._scores)
        if self._newBest:
            time_measurement.best_solution_clock = time.process_time()
        self._instances = np.concatenate((self._instances, instances))
        self._similarity_x = np.concatenate((self._similarity_x, distances), axis=None)
        self._scores = np.concatenate((self._scores, scores), axis=None)

    def __str__(self):
        achieved_indexes = self._scores > MINIMUM_SCORE
        achieved_distances = self._similarity_x[achieved_indexes]
        achieved_instances = self._instances[achieved_indexes]

        distances_counter = Counter(achieved_distances)
        sorted_distances = sorted(distances_counter.items(), key=lambda pair: pair[0], reverse=True)
        representation = []
        for k, (distance, count) in enumerate(sorted_distances):
            # find index of distance in the original array, and use that index to get the score
            index = achieved_distances.tolist().index(distance)
            instance = achieved_instances[index]
            representation.append((distance, count, instance))

        str_output = "initial instance: {}, output: {}\n".format(self._initial_instance,
                                                                 self._model.predict([self._initial_instance]))
        str_output += "Generated counterfactuals {}\n".format(self.achieved_target_count())
        for (score, count, instance) in representation:
            # str_output += "Counterfactual with score {} ({}) {}\n".format("%.4f" % score, count, instance)
            str_output += "Counterfactual with score {} ({})\n".format("%.4f" % score, count)
        return str_output

    def achieved_target(self):
        achieved_indexes = self._scores > MINIMUM_SCORE
        achieved_instances = self._instances[achieved_indexes]
        return achieved_instances

    def near(self, instance_score):
        near_best_index = self._scores > instance_score * SCORE_JITTER
        return self._instances[near_best_index]

    def info(self):
        return self._instances, self._similarity_x, self._scores

    def instances(self):
        return self._instances
