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
MINIMUM_COUNTERFACTUALS_SIZE = 100


# Remove out-of-distribution counterfactuals
def filter_outliers(counterfactuals, data_analyzer):
    print("--- Step 3: Filter outlier instances --- ")
    if not len(counterfactuals):
        print("No counterfactuals found, skipping")
        return counterfactuals
    print("Current count: {}".format(len(counterfactuals)))
    lof = LocalOutlierFactor(novelty=True)
    X, _ = data_analyzer.data()
    lof.fit(X)
    counter_pred = lof.predict(counterfactuals)
    counterfactuals = counterfactuals[counter_pred == 1]
    print("After filter: ", len(counterfactuals))
    return counterfactuals


def run(initial_instance, initial_prediction, target: Target, data_analyzer, model):
    print('--- Step 0: Load internal objects ---')

    time_measurement.init()

    score_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
    surrogate_model = RandomForestRegressor(100, n_jobs=-1, max_depth=100)
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
    instances, scores, _, _, _ = globalInstancesInfo.info()
    ranker.update(instances, scores)
    ranker.train()

    promising_instances = np.array([], dtype=np.int64).reshape(0, initial_instance.shape[0])
    best_instance, best_score, best_score_x, best_score_y, best_score_f = globalInstancesInfo.best()
    best_epoch = 0
    # --- END BOOTSTRAP ---

    iterationInstancesInfo = globalInstancesInfo
    # keep searching until good amount of counterfactuals have been found
    global_counterfactuals = globalInstancesInfo.counterfactuals()
    print('--- Step 2: Explore neighbourhood ---')
    while epoch_counter < EPOCHS_THRESHOLD and \
            (best_global_no_improvement_counter < GLOBAL_NO_IMPROVEMENT_THRESHOLD or
             len(global_counterfactuals) < MINIMUM_COUNTERFACTUALS_SIZE):
        print("--- epoch {} ----".format(epoch_counter + 1))
        # --- update counters ---
        best_global_no_improvement_counter += 1
        epoch_counter += 1
        iterations_zero_new_instances_counter += 1
        # --- end update counters ---

        instances_to_check = []
        # check if new counterfactuals have been found
        counterfactuals = iterationInstancesInfo.counterfactuals()
        if len(counterfactuals):
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
        if not len(instances_to_check) or not len(counterfactuals):
            print("No CF neighbours generated, generating random instances...")
            instances_to_check = generator.generate_random(best_instance, globalInstancesInfo.instances())
            if not len(instances_to_check):
                print("No random were generated, retrying on next epoch...")
                continue
            print("Generated random instances: ({})".format(len(instances_to_check)))

        # rank aka acquisition function
        ranked_instances = ranker.rank(globalInstancesInfo.instances(), instances_to_check)
        iterationInstancesInfo = InstancesInfo(ranked_instances, score_calculator, model)
        counterfactuals = iterationInstancesInfo.counterfactuals()
        print("Predicted top: {} Counterfactuals: {}".format(len(ranked_instances), len(counterfactuals)))

        # update known instances
        globalInstancesInfo.extend(iterationInstancesInfo)
        # update training data with known data
        instances, scores, _, _, _ = iterationInstancesInfo.info()
        ranker.update(instances, scores)

        # update new best instance, new best score, and oversample
        if globalInstancesInfo.has_new_best():
            best_instance, best_score, best_score_x, best_score_y, best_score_f = globalInstancesInfo.best()
            best_global_no_improvement_counter = 0
            best_epoch = epoch_counter
            print("Found new best {}, with fitness score {} (X {} Y {} F {}), oversampling".format(
                best_instance, "%.4f" % best_score, best_score_x, best_score_y, best_score_f)
            )
            ranker.oversample_update(best_instance, best_score)

        print("Known alternatives: {}".format(len(globalInstancesInfo)))
        print("Best instance score {} (X {} Y {} F {}), found on epoch: {}".format(
            "%.4f" % best_score, best_score_x, best_score_y, best_score_f, best_epoch)
        )
        # retrain surrogate model with updated training data
        ranker.train()
        global_counterfactuals = globalInstancesInfo.counterfactuals()

    # perform final check in instances
    print("--- Step 3: Final check on promising alternatives ({}) ---".format(len(promising_instances)))
    lastCheckInstancesInfo = InstancesInfo(promising_instances, score_calculator, model)
    counterfactuals = lastCheckInstancesInfo.counterfactuals()
    print(
        "From promising pool: ({}) Found counterfactuals: ({})".format(len(promising_instances), len(counterfactuals)))
    globalInstancesInfo.extend(lastCheckInstancesInfo)

    global_counterfactuals = globalInstancesInfo.counterfactuals()
    global_counterfactuals = filter_outliers(global_counterfactuals, data_analyzer)

    time_measurement.finish()

    return global_counterfactuals
