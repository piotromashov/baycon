import numpy as np


# distance metrics between two samples
def distance(alternative, template):
    distance = np.abs(alternative - template)
    return sum(distance)


# same as distance() applied over an array of alternatives
def distance_arr(alternatives, template):
    distance = np.abs(alternatives - template)
    return np.sum(distance, axis=1)


def not_repeated(known_instances, new_instances):
    last_idx = len(known_instances) - 1
    instances = np.concatenate((known_instances, new_instances))
    _, idx_arr = np.unique(instances, axis=0, return_index=True)
    idx_arr = idx_arr[idx_arr > last_idx]
    instances = instances[idx_arr]
    return instances


def unique_concatenate(known_instances, new_instances):
    instances = np.concatenate((known_instances, new_instances))
    return np.unique(instances, axis=0)


# returns np.array of gower distances for each instance against the initial one
def calculate_gower_distance(origin_instance, instances, features_ranges, categorical_features, weights):
    partial_gowers = np.zeros(instances.shape)
    # repeat for each column (feature values per instances)
    features_count = len(origin_instance)
    for col_idx in range(features_count):
        target = origin_instance[col_idx]
        feature_vals = instances[:, col_idx]
        feature_weight = weights[col_idx]
        # categorical or numerical, perform calculations accordingly
        if categorical_features[col_idx]:
            ij = np.where(feature_vals == target, np.zeros_like(feature_vals), np.ones_like(feature_vals))
        else:
            abs_delta = np.absolute(feature_vals - target)
            feature_range = features_ranges[col_idx]
            ij = np.divide(abs_delta, feature_range, out=np.zeros_like(abs_delta), where=feature_range != 0)
        partial_gowers[:, col_idx] = np.multiply(ij, feature_weight)

    sum_gowers = np.sum(partial_gowers, axis=1)
    gowers = np.divide(sum_gowers, weights.sum())

    return gowers