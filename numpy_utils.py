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
