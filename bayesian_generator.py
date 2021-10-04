from sklearn.ensemble import RandomForestRegressor

from InstanceInfo import InstancesInfo
from InstancesGenerator import *
from InstancesChecker import *

EPOCHS_THRESHOLD = 150  # overall number of epochs to run the algorithm
GLOBAL_NO_IMPROVEMENT_THRESHOLD = 50  # improvement on amount of epochs to stop without having improvements.


def run_generator(model, data_constraints, initial_instance, target):
    # TODO: include logging library for logging
    print('-----Starting------')
    print('model:', model, 'target:', target, 'template:', initial_instance)

    surrogate_model = RandomForestRegressor(1000, n_jobs=4)
    checker = InstancesChecker(model, surrogate_model, initial_instance, data_constraints, target)
    generator = InstancesGenerator(initial_instance, data_constraints)

    # -- COUNTERS ---
    epoch_counter = 0
    best_global_no_improvement_counter = 0
    iterations_zero_new_instances_counter = 0
    # --- END COUNTERS ---

    # --- BOOTSTRAP ---
    instances = generator.generate_initial_neighbours()
    globalInstancesInfo = InstancesInfo(instances, model, initial_instance, data_constraints, target)
    instances, distances, scores = globalInstancesInfo.info()
    checker.update(instances, scores)
    checker.train()

    promising_instances = np.array([], dtype=np.int64).reshape(0, initial_instance.shape[0])
    best_instance, best_distance, best_score = globalInstancesInfo.best()
    best_epoch = 0
    # --- END BOOTSTRAP ---

    iterationInstancesInfo = globalInstancesInfo
    while epoch_counter < EPOCHS_THRESHOLD and best_global_no_improvement_counter < GLOBAL_NO_IMPROVEMENT_THRESHOLD:
        print("--- epoch {} ----".format(epoch_counter))
        # --- update counters ---
        best_global_no_improvement_counter += 1
        epoch_counter += 1
        iterations_zero_new_instances_counter += 1
        # --- end update counters ---

        # check if there has been new positive scores
        achieved_target = iterationInstancesInfo.achieved_target_count() > 0
        if achieved_target:
            # generate neighbours if there are new scores near the current best
            instances_near_best = iterationInstancesInfo.near(best_score)
            if len(instances_near_best):
                iterations_zero_new_instances_counter = 0
                # generate neighbours near the best ones
                instances_to_check = generator.generate_neighbours_arr(instances_near_best, globalInstancesInfo.instances())
                promising_instances = npu.unique_concatenate(promising_instances, instances_to_check)
                print("Neighbours; New ({}) Promising ({})".format(len(instances_to_check),len(promising_instances)))
            else:
                instances_to_check = []

        # no new iteration instances to check, search random space within current best.
        if not len(instances_to_check) or not achieved_target:
            instances_to_check = generator.generate_random(best_distance, globalInstancesInfo.instances())
            print("No new candidates found, generated random: {}".format(len(instances_to_check)))

        # rank aka acquisition function
        ranked_instances = checker.rank(globalInstancesInfo.instances(), instances_to_check)
        iterationInstancesInfo = InstancesInfo(ranked_instances, model, initial_instance, data_constraints, target)
        print("Obtained top: {} Counterfactuals: {}".format(len(ranked_instances), iterationInstancesInfo.achieved_target_count()))

        # update known instances
        globalInstancesInfo.extend(iterationInstancesInfo)
        # update training data with known data
        instances, _, scores = iterationInstancesInfo.info()
        checker.update(instances, scores)

        # update new best instance, new best score, and oversample
        if globalInstancesInfo.has_new_best():
            best_instance, best_distance, best_score = globalInstancesInfo.best()
            best_global_no_improvement_counter = 0
            best_epoch = epoch_counter
            print("New best found {}, with {}, oversampling".format(best_instance, best_score))
            checker.oversample_update(best_instance, best_score)

        print("Known alternatives: {}".format(len(globalInstancesInfo)))
        print("Best instance: {}, score {}, found on epoch: {}".format(best_instance, best_score, best_epoch))
        # retrain surrogate model with updated training data
        checker.train()

    # perform final check in instances
    print("--- Final check on promising alternatives ---")
    lastCheckInstancesInfo = InstancesInfo(promising_instances, model, initial_instance, data_constraints, target)
    achieved_target = lastCheckInstancesInfo.achieved_target_count()
    print("Promising pool: ({}) Found counterfactuals: ({})".format(len(promising_instances), achieved_target))
    globalInstancesInfo.extend(lastCheckInstancesInfo)

    return globalInstancesInfo, 0
