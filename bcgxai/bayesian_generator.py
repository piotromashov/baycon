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

EPOCHS_THRESHOLD = 100  # overall number of epochs to run the algorithm
GLOBAL_NO_IMPROVEMENT_THRESHOLD = 10  # improvement on amount of epochs to stop without having improvements.
MINIMUM_COUNTERFACTUALS_SIZE = 100


# Remove out-of-distribution counterfactuals
def filter_outliers(counterfactuals, scores, data_analyzer):
    print("--- Step 3: Filter outlier instances --- ")
    if not len(counterfactuals):
        print("No counterfactuals found, skipping")
        return counterfactuals, scores
    print("Current count: {}".format(len(counterfactuals)))
    lof = LocalOutlierFactor(novelty=True)
    X, _ = data_analyzer.data()
    lof.fit(X)
    counter_pred = lof.predict(counterfactuals)
    counterfactuals = counterfactuals[counter_pred == 1]
    scores = scores[counter_pred == 1]
    print("After filter: ", len(counterfactuals))
    return counterfactuals, scores


def run(initial_instance, initial_prediction, target: Target, data_analyzer, model):
    print('--- Step 0: Load internal objects ---')

    time_measurement.init()

    score_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
    surrogate_model = RandomForestRegressor(1000, n_jobs=-1)
    ranker = SurrogateRanker(model, surrogate_model, initial_instance, score_calculator, target)
    generator = InstancesGenerator(initial_instance, data_analyzer, score_calculator)

    # -- COUNTERS ---
    epoch_counter = 0
    best_global_no_improvement_counter = 0
    iterations_zero_new_instances_counter = 0
    # --- END COUNTERS ---

    # --- BOOTSTRAP ---
    print('--- Step 1: Generate initial neighbours ---')
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
    # keep searching until good amount of counterfactuals have been found
    achieved_target_count = globalInstancesInfo.achieved_target_count()
    print('--- Step 2: Explore neighbourhood ---')
    while epoch_counter < EPOCHS_THRESHOLD and (
            best_global_no_improvement_counter < GLOBAL_NO_IMPROVEMENT_THRESHOLD or achieved_target_count < MINIMUM_COUNTERFACTUALS_SIZE):
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
                print("Generated neighbours: ({}) Unique overall ({})".format(len(instances_to_check),
                                                                              len(promising_instances)))
            else:
                instances_to_check = []

        # no new iteration instances to check, search random space within current best.
        if not len(instances_to_check) or not achieved_target:
            print("No CF neighbours generated, generating random instances...")
            instances_to_check = generator.generate_random(best_instance, globalInstancesInfo.instances())
            if not len(instances_to_check):
                print("No random were generated, retrying on next epoch...")
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
        print("Best instance score {}, found on epoch: {}".format("%.4f" % best_score, best_epoch))
        # retrain surrogate model with updated training data
        ranker.train()
        achieved_target_count = globalInstancesInfo.achieved_target_count()

    # perform final check in instances
    print("--- Step 3: Final check on promising alternatives ({}) ---".format(len(promising_instances)))
    lastCheckInstancesInfo = InstancesInfo(promising_instances, score_calculator, model)
    achieved_target = lastCheckInstancesInfo.achieved_target_count()
    print("From promising pool: ({}) Found counterfactuals: ({})".format(len(promising_instances), achieved_target))
    globalInstancesInfo.extend(lastCheckInstancesInfo)

    counterfactuals, scores = globalInstancesInfo.achieved_score()
    counterfactuals, scores = filter_outliers(counterfactuals, scores, data_analyzer)

    time_measurement.finish()

    return counterfactuals, scores
