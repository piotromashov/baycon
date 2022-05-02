from collections import Counter

import numpy as np

import baycon.time_measurement as time_measurement

MINIMUM_SCORE_Y = 0.1


class InstancesInfo:
    def __init__(self, instances, score_calculator, model):
        self._model = model
        self._newBest = True
        self._instances = instances
        self._scores = np.array([])
        self._scores_x = np.array([])
        self._scores_y = np.array([])
        self._scores_f = np.array([])
        self._score_calculator = score_calculator
        if not len(instances):
            return
        self.calculate_objective_all()

    def calculate_objective_all(self):
        predictions = np.array(self._model.predict(self._instances))
        self._scores, self._scores_x, self._scores_y, self._scores_f = self._score_calculator.fitness_score(
            self._instances, predictions)

        if self._scores_y[self._scores_y > MINIMUM_SCORE_Y].any():
            time_measurement.first()

    def __len__(self):
        return len(self._instances)

    def best(self):
        idx = np.argmax(self._scores)
        return self._instances[idx], self._scores[idx], self._scores_x[idx], self._scores_y[idx], self._scores_f[idx]

    def has_new_best(self):
        return self._newBest

    def achieved_target_count(self):
        return np.count_nonzero(self._scores_y > MINIMUM_SCORE_Y)

    def extend(self, instances_info):
        instances, scores, scores_x, scores_y, scores_f = instances_info.info()
        self._newBest = len(scores) and np.max(scores) > np.max(self._scores)
        if self._newBest:
            time_measurement.best()
        self._instances = np.concatenate((self._instances, instances))
        self._scores = np.concatenate((self._scores, scores), axis=None)
        self._scores_x = np.concatenate((self._scores_x, scores_x), axis=None)
        self._scores_y = np.concatenate((self._scores_y, scores_y), axis=None)
        self._scores_f = np.concatenate((self._scores_f, scores_f), axis=None)

    def __str__(self):
        achieved_indexes = self._scores_y > MINIMUM_SCORE_Y
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

        str_output = "Generated counterfactuals {}\n".format(len(self.counterfactuals()))
        for (score, count, instance) in representation:
            str_output += "Counterfactual with score {} ({})\n".format(score, count)
        return str_output

    def counterfactuals(self):
        achieved_indexes = self._scores_y > MINIMUM_SCORE_Y
        counterfactuals = self._instances[achieved_indexes]
        return counterfactuals

    def near(self, score):
        near_indexes = self._score_calculator.near_score(score, self._scores)
        return self._instances[near_indexes]

    def info(self):
        return self._instances, self._scores, self._scores_x, self._scores_y, self._scores_f

    def instances(self):
        return self._instances
