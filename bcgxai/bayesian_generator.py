import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor

import bcgxai.time_measurement as time_measurement
from bcgxai.InstancesGenerator import InstancesGenerator
from bcgxai.InstancesInfo import InstancesInfo
from bcgxai.SurrogateRanker import SurrogateRanker
from common import numpy_utils as npu
from common.ScoreCalculator import ScoreCalculator
from common.Target import *

EPOCHS_THRESHOLD = 50  # overall number of epochs to run the algorithm
GLOBAL_NO_IMPROVEMENT_THRESHOLD = 10  # improvement on amount of epochs to stop without having improvements.


# Remove out-of-distribution counterfactuals
def filter_outliers(counterfactuals, scores, data_analyzer):
    lof = LocalOutlierFactor(novelty=True)
    X, _ = data_analyzer.split_dataset()
    lof.fit(X)
    counter_pred = lof.predict(counterfactuals)
    return counterfactuals[counter_pred == 1], scores[counter_pred == 1]


def run(initial_instance, initial_prediction, target: Target, data_analyzer, model):
    print('-----Starting------')
    print('model:', model, ', target:', str(target), ', initial instance:', initial_instance)

    init_time = time.process_time()

    score_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
    surrogate_model = RandomForestRegressor(1000, n_jobs=4)
    ranker = SurrogateRanker(model, surrogate_model, initial_instance, score_calculator, target)
    generator = InstancesGenerator(initial_instance, data_analyzer, score_calculator)

    # -- COUNTERS ---
    epoch_counter = 0
    best_global_no_improvement_counter = 0
    iterations_zero_new_instances_counter = 0
    # --- END COUNTERS ---

    # --- BOOTSTRAP ---
    instances = generator.generate_initial_neighbours()
    globalInstancesInfo = InstancesInfo(instances, score_calculator, model)
    instances, scores = globalInstancesInfo.info()
    ranker.update(instances, scores)
    ranker.train()

    promising_instances = np.array([], dtype=np.int64).reshape(0, initial_instance.shape[0])
    best_instance, best_score = globalInstancesInfo.best()
    best_epoch = 0
    # --- END BOOTSTRAP ---

    iterationInstancesInfo = globalInstancesInfo
    while epoch_counter < EPOCHS_THRESHOLD and best_global_no_improvement_counter < GLOBAL_NO_IMPROVEMENT_THRESHOLD:
        print("--- epoch {} ----".format(epoch_counter + 1))
        # --- update counters ---
        best_global_no_improvement_counter += 1
        epoch_counter += 1
        iterations_zero_new_instances_counter += 1
        # --- end update counters ---

        instances_to_check = []
        # check if there has been new positive scores
        achieved_target = iterationInstancesInfo.achieved_target_count() > 0
        if achieved_target:
            # generate neighbours if there are new scores near the current best
            instances_near_best = iterationInstancesInfo.near(best_score)
            if len(instances_near_best):
                iterations_zero_new_instances_counter = 0
                # generate neighbours near the best ones
                known_instances = globalInstancesInfo.instances()
                instances_to_check = generator.generate_neighbours_arr(instances_near_best, known_instances)
                promising_instances = npu.unique_concatenate(promising_instances, instances_to_check)
                print("Neighbours; Generated ({}) Unique overall ({})".format(len(instances_to_check),
                                                                              len(promising_instances)))
            else:
                instances_to_check = []

        # no new iteration instances to check, search random space within current best.
        if not len(instances_to_check) or not achieved_target:
            print("No new candidates, generating random")
            instances_to_check = generator.generate_random(best_instance, globalInstancesInfo.instances())
            if not len(instances_to_check):
                print("No new random were generated, retrying on next epoch")
                continue
            print("Generated random instances: ({})".format(len(instances_to_check)))

        # rank aka acquisition function
        ranked_instances = ranker.rank(globalInstancesInfo.instances(), instances_to_check)
        iterationInstancesInfo = InstancesInfo(ranked_instances, score_calculator, model)
        counterfactuals = iterationInstancesInfo.achieved_target_count()
        print("Predicted top: {} Counterfactuals: {}".format(len(ranked_instances), counterfactuals))

        # update known instances
        globalInstancesInfo.extend(iterationInstancesInfo)
        # update training data with known data
        instances, scores = iterationInstancesInfo.info()
        ranker.update(instances, scores)

        # update new best instance, new best score, and oversample
        if globalInstancesInfo.has_new_best():
            best_instance, best_score = globalInstancesInfo.best()
            best_global_no_improvement_counter = 0
            best_epoch = epoch_counter
            print("New best found {}, with {}, oversampling".format(best_instance, "%.4f" % best_score))
            ranker.oversample_update(best_instance, best_score)

        print("Known alternatives: {}".format(len(globalInstancesInfo)))
        print("Best instance: {}, score {}, found on epoch: {}".format(best_instance, "%.4f" % best_score, best_epoch))
        # retrain surrogate model with updated training data
        ranker.train()

    # perform final check in instances
    print("--- Final check on promising alternatives ---")
    lastCheckInstancesInfo = InstancesInfo(promising_instances, score_calculator, model)
    achieved_target = lastCheckInstancesInfo.achieved_target_count()
    print("Promising pool: ({}) Found counterfactuals: ({})".format(len(promising_instances), achieved_target))
    globalInstancesInfo.extend(lastCheckInstancesInfo)

    counterfactuals, scores = globalInstancesInfo.achieved_score()
    print("Before filter: ", len(counterfactuals))
    counterfactuals, scores = filter_outliers(counterfactuals, scores, data_analyzer)
    print("After filter: ", len(counterfactuals))

    time_measurement.time_to_first_solution = time_measurement.first_solution_clock - init_time
    time_measurement.time_to_best_solution = time_measurement.best_solution_clock - init_time
    time_measurement.total_time = time.process_time() - init_time

    return counterfactuals, scores
