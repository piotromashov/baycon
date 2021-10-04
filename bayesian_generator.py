from sklearn.ensemble import RandomForestRegressor

from InstanceInfo import InstancesInfo
from InstancesGenerator import *
from InstancesChecker import *


def run_generator(model, dataconstraints, initial_instance, target):
    # TODO: include logging library for log
    print('-----Starting------')
    print('model:', model, 'target:', target, 'template:', initial_instance)

    # TODO: read from config file
    # --- CONFIG ---
    random_sample_size = 10000  # random instances for the bayesian model
    oversampling_weight = 10  # TODO should we keep oversampling?
    neighbours_max_degree = 3
    neighborhood_jitter = .75  # giving search space for the solutions we are finding
    changes_jitter = 8  # TODO: put 20% of feature size

    epochs_threshold = 150  # overall number of epochs to run the algorithm
    best_global_no_improvement_threshold = 50  # improvement on amount of epochs to stop without having improvements.

    TOP_RANKED = 20
    # --- END CONFIG ---

    # -- COUNTERS ---
    epoch_counter = 0
    best_global_no_improvement_counter = 0
    iterations_zero_new_instances_counter = 0
    epoch_max_found = 0
    # --- END COUNTERS ---

    surrogate_model = RandomForestRegressor(1000, n_jobs=4)
    checker = InstancesChecker(model, surrogate_model, initial_instance, dataconstraints, target)
    generator = InstancesGenerator(initial_instance, dataconstraints, neighbours_max_degree)

    # --- BOOTSTRAP ---
    # TODO here we should generate the neighbours of the initial instance
    instances = generator.generate_initial_neighbours()
    globalInstancesInfo = InstancesInfo(instances, model, initial_instance, dataconstraints, target)
    X, _, Y = globalInstancesInfo.info()
    # TODO improvement: oversample X based on score Y
    checker.train_surrogate(X, Y)
    # --- END BOOTSTRAP ---

    promising_alternatives_pool = np.array([], dtype=np.int64).reshape(0, initial_instance.shape[0])
    epoch_max_found = 0

    # stoh_start_time = time.time()
    # stoh_duration =  time.time() - stoh_start_time  #time to first solution

    iterationInstancesInfo = globalInstancesInfo
    while epoch_counter < epochs_threshold and best_global_no_improvement_counter < best_global_no_improvement_threshold:
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
            _, _, best_score = globalInstancesInfo.best()
            instances_near_best = iterationInstancesInfo.near(best_score, neighborhood_jitter)
            if len(instances_near_best):
                iterations_zero_new_instances_counter = 0
                # generate neighbours near the best ones
                instances = generator.generate_neighbours_arr(instances_near_best, globalInstancesInfo.instances())
                promising_alternatives_pool = npu.unique_concatenate(promising_alternatives_pool, instances)
                print("Found new neighbours: {}".format(len(instances)))
            else:
                instances = []

        # no new iteration instances to check, search random space within current best.
        if not len(instances) or not achieved_target:
            _, bestDistance, _ = globalInstancesInfo.best()
            num_changes = bestDistance * changes_jitter
            instances = generator.generate_random(random_sample_size, num_changes, globalInstancesInfo.instances())
            print("No new candidates found, generated random: {}".format(len(instances)))

        # rank aka acquisition function
        instances = checker.rank(X, instances, TOP_RANKED)
        print("Obtained top {}".format(len(instances)))
        iterationInstancesInfo = InstancesInfo(instances, model, initial_instance, dataconstraints, target)
        print("Out if which, counterfactuals are: {}".format(iterationInstancesInfo.achieved_target_count()))

        # update known instances
        globalInstancesInfo.extend(iterationInstancesInfo)
        # update training data with known data
        instances, _, scores = iterationInstancesInfo.info()
        X = np.concatenate((X, instances))
        Y = np.concatenate((Y, scores), axis=None)

        # update new best instance and new best score
        # TODO: improvement: update training data with oversampled instances based on scores
        best_instance, _, best_score = globalInstancesInfo.best()
        if globalInstancesInfo.has_new_best():
            best_global_no_improvement_counter = 0
            epoch_max_found = epoch_counter
            # oversample new best
            print("New best found {}, with {}, oversampling".format(best_instance, best_score))
            # TODO: oversample_update on checker object
            oversampled_best_instance = np.repeat([best_instance], oversampling_weight, axis=0)
            oversampled_best_score = np.repeat(best_score, oversampling_weight)
            # update training data with oversampled best
            X = np.concatenate((X, oversampled_best_instance))
            Y = np.concatenate((Y, oversampled_best_score), axis=None)

        print("Known alternatives: {}".format(len(globalInstancesInfo)))
        print("Promising alternatives: {}".format(len(promising_alternatives_pool)))
        print("Best instance: {}, score {}, found on epoch: {}".format(best_instance, best_score, epoch_max_found))

        # retrain surrogate model with updated training data
        # TODO: improvement: do this if there has been an % increase in data
        print("Re-training surrogate model with data size: {}".format(X.shape[0]))
        checker.train_surrogate(X, Y)

    # perform final check in instances
    print("--- Final check on promising alternatives ---")
    lastCheckInstancesInfo = InstancesInfo(promising_alternatives_pool, model, initial_instance, dataconstraints, target)
    achieved_target = lastCheckInstancesInfo.achieved_target_count()
    print("Promising pool: ({}), found counterfactuals: ({})".format(len(promising_alternatives_pool), achieved_target))
    globalInstancesInfo.extend(lastCheckInstancesInfo)

    return globalInstancesInfo, 0
