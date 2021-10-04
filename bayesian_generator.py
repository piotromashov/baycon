from sklearn.ensemble import RandomForestRegressor
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
    best_global_no_improvement_threshold = 50  # improvement on amout of epochs to stop without having improvements.

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
    X = generator.generate_initial_neighbours()
    Y = checker.calculate_objective_all(X)
    # TODO improvement: oversample X based on score Y
    checker.train_surrogate(X, Y)
    # --- END BOOTSTRAP ---

    all_counterfactuals = X[Y > 0]
    all_counterfactuals_scores = Y[Y > 0]
    known_instances = X.copy()  # known_alternatives to avoid duplicates
    promising_alternatives_pool = np.array([], dtype=np.int64).reshape(0, initial_instance.shape[0])
    iteration_instances = X
    iteration_scores = Y

    best_global_instance = X[np.argmax(Y)]
    best_global_score = max(Y)
    epoch_max_found = 0

    # stoh_start_time = time.time()
    # stoh_duration =  time.time() - stoh_start_time  #time to first solution

    while epoch_counter < epochs_threshold and best_global_no_improvement_counter < best_global_no_improvement_threshold:
        print("--- epoch {} ----".format(epoch_counter))
        # --- update counters ---
        best_global_no_improvement_counter += 1
        epoch_counter += 1
        iterations_zero_new_instances_counter += 1
        # --- end update counters ---

        # check if there has been new positive scores
        scores_achieved_target = iteration_scores[iteration_scores > 0.01]
        if len(scores_achieved_target):

            # generate neighbours if there are new scores near the current best
            scores_near_best_index = iteration_scores > best_global_score * neighborhood_jitter
            scores_near_best = iteration_scores[scores_near_best_index]
            if len(scores_near_best):
                iterations_zero_new_instances_counter = 0

                # generate neighbours near the best ones
                instances_near_best = iteration_instances[scores_near_best_index]
                iteration_instances = generator.generate_neighbours_arr(instances_near_best, known_instances)
                promising_alternatives_pool = npu.unique_concatenate(promising_alternatives_pool, iteration_instances)
                print("Found new neighbours: {}".format(len(iteration_instances)))
            else:
                iteration_instances = []

        # no new iteration instances to check, search random space within current best.
        if not len(iteration_instances) or not len(scores_achieved_target):
            distance_from_best_to_initial_instance = npu.distance(np.array(best_global_instance),
                                                                  np.array(initial_instance))
            num_changes = distance_from_best_to_initial_instance * changes_jitter
            iteration_instances = generator.generate_random(random_sample_size, num_changes, known_instances)
            print("No new candidates found, generated random: {}".format(len(iteration_instances)))

        # rank aka acquisition function
        iteration_instances = checker.rank(X, iteration_instances, TOP_RANKED)
        print("Obtained top {}".format(len(iteration_instances)))
        iteration_scores = checker.calculate_objective_all(iteration_instances)

        # update new counterfactuals found
        top_scores_achieved_target_index = iteration_scores > 0.01
        counterfactuals = iteration_instances[top_scores_achieved_target_index]
        print("Out if which, counterfactuals are: {}".format(len(counterfactuals)))
        all_counterfactuals = np.concatenate((all_counterfactuals, counterfactuals))
        top_scores_achieved_target = iteration_scores[top_scores_achieved_target_index]
        all_counterfactuals_scores = np.concatenate((all_counterfactuals_scores, top_scores_achieved_target))

        # update known instances
        known_instances = npu.unique_concatenate(known_instances, iteration_instances)
        # update training data with known data
        X = np.concatenate((X, iteration_instances))
        Y = np.concatenate((Y, iteration_scores), axis=None)

        # update new best instance and new best score
        # TODO: improvement: update training data with oversampled instances based on scores
        new_best_global_score_index = iteration_scores > best_global_score
        new_best_global_score = iteration_scores[new_best_global_score_index]
        if len(new_best_global_score):
            best_global_no_improvement_counter = 0
            epoch_max_found = epoch_counter
            best_global_score = new_best_global_score[0]
            best_global_instance = iteration_instances[new_best_global_score_index][0]
            # oversample new best
            print("New best found {}, with {}, oversampling".format(best_global_instance, best_global_score))
            oversampled_best_instance = np.repeat([best_global_instance], oversampling_weight, axis=0)
            oversampled_best_score = np.repeat(best_global_score, oversampling_weight)
            # update training data with oversampled best
            X = np.concatenate((X, oversampled_best_instance))
            Y = np.concatenate((Y, oversampled_best_score), axis=None)

        print("Known alternatives: {}".format(len(known_instances)))
        print("Promising alternatives: {}".format(len(promising_alternatives_pool)))
        print("Best instance: {}, score {}, found on epoch: {}".format(best_global_instance, best_global_score,
                                                                       epoch_max_found))

        # retrain surrogate model with updated training data
        # TODO: improvement: do this if there has been an % increase in data
        print("Re-training surrogate model with data size: {}".format(X.shape[0]))
        checker.train_surrogate(X, Y)

    # perform final check in instances
    print("--- Final check on promising alternatives ---")
    max_iterations = 1
    i = 0
    while len(promising_alternatives_pool) and i < max_iterations:
        i += 1
        promising_alternatives_scores = checker.calculate_objective_all(promising_alternatives_pool)
        known_instances = npu.unique_concatenate(known_instances, promising_alternatives_pool)
        promising_alternatives_achieved_target_index = promising_alternatives_scores > 0.01
        # update counterfactuals
        counterfactuals = promising_alternatives_pool[promising_alternatives_achieved_target_index]
        counterfactuals_scores = promising_alternatives_scores[promising_alternatives_achieved_target_index]
        print("From promising pool: {}, found counterfactuals: {}".format(len(promising_alternatives_pool),
                                                                          len(counterfactuals)))
        all_counterfactuals = np.concatenate((all_counterfactuals, counterfactuals))
        all_counterfactuals_scores = np.concatenate((all_counterfactuals_scores, counterfactuals_scores))
        # promising_alternatives_pool = generator.generate_neighbours_arr(counterfactuals, known_instances)

    print("Generated counterfactuals {}".format(len(all_counterfactuals)))
    return all_counterfactuals, all_counterfactuals_scores, 0
