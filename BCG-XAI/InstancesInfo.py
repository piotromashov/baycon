import time
from collections import Counter

import numpy as np

import time_measurement

MINIMUM_SCORE = 0


class InstancesInfo:
    def __init__(self, instances, similarity_calculator, model):
        self._model = model
        self._newBest = True
        self._instances = instances
        self._scores = []
        self._similarity_calculator = similarity_calculator
        self.calculate_objective_all()

    def calculate_objective_all(self):
        predictions = np.array(self._model.predict(self._instances))
        self._scores = self._similarity_calculator.calculate_scores(self._instances, predictions)

        if self._scores[self._scores > MINIMUM_SCORE].any() and not time_measurement.first_solution_clock:
            time_measurement.first_solution_clock = time.process_time()

    def __len__(self):
        return len(self._instances)

    def best(self):
        index = np.argmax(self._scores)
        return self._instances[index], self._scores[index]

    def has_new_best(self):
        return self._newBest

    def achieved_target_count(self):
        return np.count_nonzero(self._scores > MINIMUM_SCORE)

    def extend(self, instances_info):
        instances, scores = instances_info.info()
        self._newBest = np.max(scores) > np.max(self._scores)
        if self._newBest:
            time_measurement.best_solution_clock = time.process_time()
        self._instances = np.concatenate((self._instances, instances))
        self._scores = np.concatenate((self._scores, scores), axis=None)

    def __str__(self):
        achieved_indexes = self._scores > MINIMUM_SCORE
        achieved_scores = np.round(self._scores[achieved_indexes], 2)
        achieved_instances = self._instances[achieved_indexes]

        scores_counter = Counter(achieved_scores)
        sorted_scores = sorted(scores_counter.items(), key=lambda pair: pair[0], reverse=True)
        representation = []
        for k, (score, count) in enumerate(sorted_scores):
            # find index of distance in the original array, and use that index to get the score
            index = achieved_scores.tolist().index(score)
            instance = achieved_instances[index]
            representation.append((score, count, instance))

        str_output = "Generated counterfactuals {}\n".format(self.achieved_target_count())
        for (score, count, instance) in representation:
            str_output += "Counterfactual with score {} ({})\n".format(score, count)
        return str_output

    def achieved_target(self):
        achieved_indexes = self._scores > MINIMUM_SCORE
        achieved_instances = self._instances[achieved_indexes]
        return achieved_instances

    def near(self, score):
        indexes = self._similarity_calculator.near_similarity(score, self._scores)
        return self._instances[indexes]

    def info(self):
        return self._instances, self._scores

    def instances(self):
        return self._instances
